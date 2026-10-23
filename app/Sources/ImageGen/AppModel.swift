import AppKit
import Foundation
import Observation

@MainActor
@Observable
final class AppModel {
    enum Phase: Equatable {
        case starting(String)
        case failed(String)
        case running
    }

    // Engine connection
    var phase: Phase = .starting("Starting…")
    var engine = EngineState(modelId: nil, status: "unloaded", message: "No model loaded", device: nil)
    var mcpURL = ""
    var outputDir = ""
    var freeDiskGb: Double = 0
    var logLines: [String] = []

    // Data
    var models: [ModelInfo] = []
    var jobs: [Job] = []
    var history: [GeneratedImage] = []

    // Navigation
    var showModelPicker = true
    var selectedImage: GeneratedImage? {
        didSet { showsJobProgress = false }
    }
    /// The canvas shows the running job instead of `selectedImage`.
    var showsJobProgress = false
    var errorMessage: String?

    // Generation form
    var prompt = ""
    var negativePrompt = ""
    var aspectRatio = "1:1"
    var quality: Quality = .standard
    var steps: Int = 40
    var useCustomSteps = false
    var seedText = ""
    var cfgScale: Double = 1.0
    var transparent = false
    var referenceImages: [URL] = []

    enum Quality: String, CaseIterable, Identifiable {
        case draft, standard, high
        var id: String { rawValue }
        var preferredResolution: Int { [.draft: 512, .standard: 1024, .high: 2048][self]! }
        var label: String { rawValue.capitalized }

        /// "High" shrinks on Macs that can't fit native 2048 px in memory.
        func resolution(max: Int?) -> Int { min(preferredResolution, max ?? 2048) }

        func detail(max: Int?) -> String {
            switch self {
            case .draft: return "~512 px, fastest"
            case .standard: return "~1 MP"
            case .high:
                let res = resolution(max: max)
                let mp = String(format: "%.1f", Double(res * res) / 1_000_000)
                return res < 2048 ? "~\(mp) MP, the most this Mac's memory allows" : "~\(mp) MP, native, slowest"
            }
        }
    }

    @ObservationIgnored private let process = EngineProcess()
    @ObservationIgnored private var pollTask: Task<Void, Never>?
    @ObservationIgnored private var knownFinishedJobs = Set<String>()
    @ObservationIgnored private var lastHistoryRefresh = Date.distantPast

    var port: Int { UserDefaults.standard.integer(forKey: "port").nonZero ?? 7860 }
    var client: APIClient { APIClient(baseURL: URL(string: "http://127.0.0.1:\(port)")!) }
    var lastModelId: String? {
        get { UserDefaults.standard.string(forKey: "lastModelId") }
        set { UserDefaults.standard.set(newValue, forKey: "lastModelId") }
    }

    var currentModel: ModelInfo? { models.first { $0.id == engine.modelId } }
    var activeJobs: [Job] { jobs.filter(\.isActive).sorted { $0.createdAt < $1.createdAt } }
    var claudeCommand: String { "claude mcp add --transport http imagegen \(mcpURL)" }
    var physicalMemoryGb: Int { Int(ProcessInfo.processInfo.physicalMemory / 1_073_741_824) }

    // MARK: Engine lifecycle

    func start() {
        guard pollTask == nil else { return }
        Task { await boot() }
    }

    func restart() {
        pollTask?.cancel()
        pollTask = nil
        process.stop()
        phase = .starting("Restarting…")
        Task { await boot() }
    }

    func shutdown() {
        pollTask?.cancel()
        process.stop()
    }

    private func boot() async {
        process.onLog = { [weak self] text in
            Task { @MainActor in self?.appendLog(text) }
        }
        do {
            if await client.health() {
                appendLog("Connected to an engine that is already running on port \(port).\n")
            } else {
                try await process.start(port: port, hfToken: Keychain.hfToken) { [weak self] message in
                    Task { @MainActor in self?.phase = .starting(message) }
                }
            }
            await refresh(includeModels: true, includeHistory: true)
            if ProcessInfo.processInfo.environment["IMAGEGEN_SKIP_PICKER"] != nil, engine.isReady { showModelPicker = false }
            phase = .running
            startPolling()
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func appendLog(_ text: String) {
        logLines.append(contentsOf: text.split(separator: "\n", omittingEmptySubsequences: true).map(String.init))
        if logLines.count > 2000 { logLines.removeFirst(logLines.count - 2000) }
    }

    private func startPolling() {
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let busy = self.activeJobs.isEmpty == false || self.models.contains { $0.download?.isActive == true } || self.engine.isLoading
                await self.refresh(includeModels: self.showModelPicker || busy, includeHistory: false)
                try? await Task.sleep(for: .milliseconds(busy ? 500 : 1500))
            }
        }
    }

    func refresh(includeModels: Bool, includeHistory: Bool) async {
        do {
            let status = try await client.status()
            let wasLoading = engine.isLoading
            engine = status.engine
            jobs = status.jobs
            mcpURL = status.mcpUrl
            outputDir = status.outputDir
            freeDiskGb = status.freeDiskGb
            if wasLoading && !engine.isLoading { await refreshModels() }

            // A job finished (from the UI or via MCP): refresh history and show the new image.
            let finished = status.jobs.filter { $0.status == "done" }
            let fresh = finished.filter { !knownFinishedJobs.contains($0.id) }
            fresh.forEach { knownFinishedJobs.insert($0.id) }
            if !fresh.isEmpty || includeHistory || Date().timeIntervalSince(lastHistoryRefresh) > 10 {
                await refreshHistory()
                if let newest = fresh.max(by: { $0.createdAt < $1.createdAt })?.image, lastHistoryRefresh != .distantPast {
                    selectedImage = history.first { $0.name == newest.name } ?? newest
                }
            }
            if includeModels { await refreshModels() }
        } catch {
            if !process.isRunning, case .running = phase, !(await client.health()) {
                phase = .failed("Lost connection to the engine.")
                pollTask?.cancel()
                pollTask = nil
            }
        }
    }

    private func refreshModels() async {
        if let models = try? await client.models() { self.models = models }
    }

    private func refreshHistory() async {
        if let history = try? await client.history() {
            self.history = history
            if let selected = selectedImage, !history.contains(selected) {
                selectedImage = history.first { $0.name == selected.name }
            }
        }
        if lastHistoryRefresh == .distantPast, selectedImage == nil { selectedImage = history.first }
        lastHistoryRefresh = Date()
    }

    // MARK: Models

    func download(_ model: ModelInfo) {
        run { try await self.client.download(model.id) }
    }

    func cancelDownload(_ model: ModelInfo) {
        run { try await self.client.cancelDownload(model.id) }
    }

    func delete(_ model: ModelInfo) {
        run { try await self.client.deleteModel(model.id) }
    }

    func use(_ model: ModelInfo) {
        lastModelId = model.id
        applyDefaults(for: model)
        showModelPicker = false
        run { try await self.client.load(model.id) }
    }

    private func applyDefaults(for model: ModelInfo) {
        steps = model.defaultSteps
        if !model.aspectRatios.contains(aspectRatio) { aspectRatio = model.aspectRatios.first ?? "1:1" }
    }

    // MARK: Generation

    var canGenerate: Bool {
        !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && (engine.isReady || engine.isLoading)
    }

    func generate() {
        guard canGenerate else { return }
        let request = GenerateRequest(
            prompt: prompt,
            negativePrompt: negativePrompt.isEmpty ? nil : negativePrompt,
            aspectRatio: aspectRatio,
            resolution: quality.resolution(max: engine.maxResolution),
            steps: useCustomSteps && currentModel?.fixedSteps != true ? steps : nil,
            seed: Int(seedText),
            cfgScale: negativePrompt.isEmpty ? 1.0 : cfgScale,
            transparent: transparent,
            referenceImages: referenceImages.map(\.path)
        )
        run {
            let job = try await self.client.generate(request)
            self.jobs.insert(job, at: 0)
            self.showsJobProgress = true
        }
    }

    func cancel(_ job: Job) {
        run { try await self.client.cancelJob(job.id) }
    }

    func reuseSettings(of image: GeneratedImage) {
        prompt = image.prompt
        negativePrompt = image.negativePrompt ?? ""
        aspectRatio = image.aspectRatio ?? aspectRatio
        transparent = image.transparent ?? false
        seedText = image.seed.map(String.init) ?? ""
        if let steps = image.steps { self.steps = steps; useCustomSteps = steps != currentModel?.defaultSteps }
        if let res = image.resolution, let q = Quality.allCases.first(where: { $0.resolution(max: engine.maxResolution) == res }) { quality = q }
        referenceImages = (image.referenceImages ?? []).map { URL(fileURLWithPath: $0) }
    }

    func addReference(_ url: URL) {
        let limit = currentModel?.maxReferenceImages ?? 10
        guard !referenceImages.contains(url), referenceImages.count < limit else { return }
        referenceImages.append(url)
    }

    func delete(_ image: GeneratedImage) {
        run {
            try await self.client.deleteImage(image.name)
            if let index = self.history.firstIndex(of: image) {
                self.history.remove(at: index)
                if self.selectedImage == image {
                    self.selectedImage = self.history.indices.contains(index) ? self.history[index] : self.history.last
                }
            }
        }
    }

    private func run(_ action: @escaping @MainActor () async throws -> Void) {
        Task {
            do {
                try await action()
                await refresh(includeModels: true, includeHistory: false)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}

extension Int {
    var nonZero: Int? { self == 0 ? nil : self }
}
