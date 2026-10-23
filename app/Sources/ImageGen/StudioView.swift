import SwiftUI
import UniformTypeIdentifiers

struct StudioView: View {
    @Environment(AppModel.self) private var model
    @State private var showInspector = true
    @State private var showMCP = false

    var body: some View {
        NavigationSplitView {
            ControlsPanel()
                .navigationSplitViewColumnWidth(min: 290, ideal: 320, max: 420)
        } detail: {
            VStack(spacing: 0) {
                CanvasView()
                Divider()
                HistoryStrip()
            }
            .inspector(isPresented: $showInspector) {
                ImageInspector()
                    .inspectorColumnWidth(min: 240, ideal: 280, max: 360)
            }
        }
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button {
                    model.showModelPicker = true
                } label: {
                    EngineBadge()
                }
            }
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    showMCP.toggle()
                } label: {
                    Label("MCP", systemImage: "point.3.connected.trianglepath.dotted")
                }
                .help("Connect Claude Code or another MCP client")
                .popover(isPresented: $showMCP, arrowEdge: .bottom) { MCPPanel() }

                Button {
                    NSWorkspace.shared.open(URL(fileURLWithPath: model.outputDir))
                } label: {
                    Label("Open Output Folder", systemImage: "folder")
                }
                .help("Open output folder")

                Button {
                    showInspector.toggle()
                } label: {
                    Label("Inspector", systemImage: "sidebar.trailing")
                }
                .help("Show details")
            }
        }
    }
}

private struct EngineBadge: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(model.currentModel?.name ?? "No model")
                .fontWeight(.medium)
            if model.engine.isLoading {
                ProgressView().controlSize(.mini)
            }
        }
        .padding(.horizontal, 4)
        .help(helpText)
    }

    private var helpText: String {
        guard let gb = model.engine.gpuMemoryGb, model.engine.isReady else { return model.engine.message }
        let limit = model.engine.gpuMemoryLimitGb.map { " of \(String(format: "%.0f", $0))" } ?? ""
        return "\(model.engine.message) · \(String(format: "%.1f", gb))\(limit) GB GPU memory in use. Click to switch model."
    }

    private var color: Color {
        switch model.engine.status {
        case "ready": .green
        case "loading": .orange
        case "error": .red
        default: .secondary
        }
    }
}

// MARK: - Controls

private struct ControlsPanel: View {
    @Environment(AppModel.self) private var model
    @State private var showAdvanced = false
    @FocusState private var promptFocused: Bool

    var body: some View {
        @Bindable var model = model
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    section("Prompt") {
                        TextEditor(text: $model.prompt)
                            .font(.body)
                            .scrollContentBackground(.hidden)
                            .padding(6)
                            .frame(minHeight: 110, maxHeight: 220)
                            .background(.background, in: RoundedRectangle(cornerRadius: 8))
                            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.separator))
                            .focused($promptFocused)
                            .overlay(alignment: .topLeading) {
                                if model.prompt.isEmpty {
                                    Text(model.referenceImages.isEmpty
                                         ? "Describe the image. Put text that should appear in \"quotes\"."
                                         : "Describe the edit, e.g. \"Change the background to a sunset beach\".")
                                        .foregroundStyle(.tertiary)
                                        .padding(.horizontal, 11)
                                        .padding(.vertical, 6)
                                        .allowsHitTesting(false)
                                }
                            }
                    }

                    section("Aspect ratio") {
                        AspectRatioPicker(selection: $model.aspectRatio, ratios: model.currentModel?.aspectRatios ?? ["1:1"])
                            .disabled(!model.referenceImages.isEmpty)
                        if !model.referenceImages.isEmpty {
                            Text("Edits keep the aspect ratio of the last reference image.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }

                    section("Quality") {
                        Picker("Quality", selection: $model.quality) {
                            ForEach(AppModel.Quality.allCases) { Text($0.label).tag($0) }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        Text(model.quality.detail(max: model.engine.maxResolution)).font(.caption).foregroundStyle(.secondary)
                    }

                    Toggle(isOn: $model.transparent) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Transparent background")
                            Text("RGBA output for stickers, icons and cutouts").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .toggleStyle(.switch)

                    section("Reference images") {
                        ReferenceImagesView()
                    }

                    DisclosureGroup("Advanced", isExpanded: $showAdvanced) {
                        VStack(alignment: .leading, spacing: 12) {
                            if model.currentModel?.fixedSteps == true {
                                Text("This model uses a fixed \(model.currentModel?.defaultSteps ?? 8)-step sampling schedule.")
                                    .font(.caption).foregroundStyle(.secondary)
                            } else {
                                Toggle("Custom steps", isOn: $model.useCustomSteps)
                            }
                            if model.useCustomSteps && model.currentModel?.fixedSteps != true {
                                HStack {
                                    Slider(value: Binding(get: { Double(model.steps) }, set: { model.steps = Int($0) }), in: 4...80, step: 1)
                                    Text("\(model.steps)").monospacedDigit().frame(width: 28, alignment: .trailing)
                                }
                            }
                            TextField("Seed (random if empty)", text: $model.seedText)
                                .textFieldStyle(.roundedBorder)
                            TextField("Negative prompt", text: $model.negativePrompt, axis: .vertical)
                                .textFieldStyle(.roundedBorder)
                                .lineLimit(1...4)
                            if !model.negativePrompt.isEmpty {
                                HStack {
                                    Text("Guidance")
                                    Slider(value: $model.cfgScale, in: 1...8, step: 0.5)
                                    Text(String(format: "%.1f", model.cfgScale)).monospacedDigit().frame(width: 28)
                                }
                                Text("The negative prompt only applies when guidance is above 1. It doubles the time per step.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        .padding(.top, 8)
                    }
                }
                .padding(16)
            }

            Divider()
            VStack(spacing: 8) {
                Button {
                    model.generate()
                } label: {
                    Label(model.referenceImages.isEmpty ? "Generate" : "Edit", systemImage: "sparkles")
                        .frame(maxWidth: .infinity)
                }
                .controlSize(.large)
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!model.canGenerate)
                Text(footnote).font(.caption).foregroundStyle(.secondary)
            }
            .padding(16)
        }
        .onAppear { promptFocused = true }
    }

    private var footnote: String {
        if model.engine.isLoading { return "Model is loading. Jobs will start when it is ready." }
        if !model.engine.isReady { return model.engine.message }
        let queued = model.activeJobs.count
        return queued > 0 ? "\(queued) job\(queued == 1 ? "" : "s") in queue · ⌘↩" : "⌘↩ to generate"
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.subheadline.weight(.semibold)).foregroundStyle(.secondary)
            content()
        }
    }
}

private struct AspectRatioPicker: View {
    @Binding var selection: String
    let ratios: [String]

    var body: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 4), spacing: 6) {
            ForEach(ratios, id: \.self) { ratio in
                Button {
                    selection = ratio
                } label: {
                    VStack(spacing: 4) {
                        RatioShape(ratio: ratio)
                            .frame(height: 22)
                        Text(ratio).font(.caption.monospacedDigit())
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
                    .background(selection == ratio ? Color.accentColor.opacity(0.18) : Color.clear, in: RoundedRectangle(cornerRadius: 7))
                    .overlay(RoundedRectangle(cornerRadius: 7).strokeBorder(selection == ratio ? Color.accentColor : Color.secondary.opacity(0.25)))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
    }
}

private struct RatioShape: View {
    let ratio: String

    var body: some View {
        let parts = ratio.split(separator: ":").compactMap { Double($0) }
        let r = parts.count == 2 ? parts[0] / parts[1] : 1
        let w = r >= 1 ? 22.0 : 22.0 * r
        let h = r >= 1 ? 22.0 / r : 22.0
        RoundedRectangle(cornerRadius: 2)
            .strokeBorder(lineWidth: 1.5)
            .frame(width: w, height: h)
            .frame(width: 22, height: 22)
    }
}

private struct ReferenceImagesView: View {
    @Environment(AppModel.self) private var model
    @State private var targeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.referenceImages.isEmpty {
                LazyVGrid(columns: Array(repeating: GridItem(.fixed(58), spacing: 6), count: 4), alignment: .leading, spacing: 6) {
                    ForEach(Array(model.referenceImages.enumerated()), id: \.element) { index, url in
                        Thumbnail(path: url.path, size: 58)
                            .frame(width: 58, height: 58)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(alignment: .topTrailing) {
                                Button {
                                    model.referenceImages.removeAll { $0 == url }
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .symbolRenderingMode(.palette)
                                        .foregroundStyle(.white, .black.opacity(0.6))
                                }
                                .buttonStyle(.plain)
                                .padding(3)
                            }
                            .overlay(alignment: .bottomLeading) {
                                Text("\(index + 1)")
                                    .font(.caption2.bold())
                                    .padding(.horizontal, 4)
                                    .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 3))
                                    .foregroundStyle(.white)
                                    .padding(3)
                            }
                            .help(url.lastPathComponent)
                    }
                }
            }
            Button {
                pickFiles()
            } label: {
                VStack(spacing: 4) {
                    Image(systemName: "photo.badge.plus").font(.title3)
                    Text(model.referenceImages.isEmpty ? "Drop images to edit or combine" : "Add more")
                        .font(.caption)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .foregroundStyle(targeted ? Color.accentColor : .secondary)
                .background(targeted ? Color.accentColor.opacity(0.1) : .clear, in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [5])).foregroundStyle(.secondary.opacity(0.5)))
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .dropDestination(for: URL.self) { urls, _ in
                urls.filter(\.isFileURL).forEach(model.addReference)
                return true
            } isTargeted: { targeted = $0 }
        }
    }

    private func pickFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK { panel.urls.forEach(model.addReference) }
    }
}

// MARK: - Canvas

private struct CanvasView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ZStack {
            Color(nsColor: .underPageBackgroundColor)
            if let job = model.activeJobs.first(where: { $0.status == "running" }) ?? model.activeJobs.first,
               model.selectedImage == nil || model.showsJobProgress {
                JobProgressCard(job: job)
            } else if let image = model.selectedImage {
                ImageViewer(image: image)
            } else {
                EmptyCanvas()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct EmptyCanvas: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "sparkles.rectangle.stack")
                .font(.system(size: 48, weight: .light))
                .foregroundStyle(.tertiary)
            Text(model.engine.isLoading ? model.engine.message : "Write a prompt and press ⌘↩")
                .font(.title3)
                .foregroundStyle(.secondary)
            if model.engine.isLoading {
                ProgressView().controlSize(.small)
                Text("Loading a large model takes a minute or two.").font(.callout).foregroundStyle(.tertiary)
            } else {
                Text("Claude Code and other MCP clients can generate here too.").font(.callout).foregroundStyle(.tertiary)
            }
        }
    }
}

private struct JobProgressCard: View {
    @Environment(AppModel.self) private var model
    let job: Job

    var body: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle().stroke(.quaternary, lineWidth: 8)
                Circle()
                    .trim(from: 0, to: job.fraction)
                    .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut, value: job.fraction)
                Text("\(Int(job.fraction * 100))%").font(.title2.monospacedDigit().weight(.medium))
            }
            .frame(width: 110, height: 110)

            VStack(spacing: 4) {
                Text(job.stage).font(.headline)
                if let started = job.startedAt, job.step > 0 {
                    let elapsed = Date().timeIntervalSince1970 - started
                    let perStep = elapsed / Double(job.step)
                    Text("\(elapsed.durationString) elapsed · about \((perStep * Double(job.totalSteps - job.step)).durationString) left")
                        .font(.callout).foregroundStyle(.secondary).monospacedDigit()
                }
                if job.source == "mcp" {
                    Label("Requested via MCP", systemImage: "point.3.connected.trianglepath.dotted")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Text(job.params.prompt)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .lineLimit(3)
                .frame(maxWidth: 460)
            Button("Cancel") { model.cancel(job) }
            if model.activeJobs.count > 1 {
                Text("\(model.activeJobs.count - 1) more waiting").font(.caption).foregroundStyle(.tertiary)
            }
        }
        .padding(28)
    }
}

private struct ImageViewer: View {
    let image: GeneratedImage
    @State private var nsImage: NSImage?

    var body: some View {
        GeometryReader { geo in
            let fitted = fittedSize(in: CGSize(width: geo.size.width - 40, height: geo.size.height - 40))
            ZStack {
                if image.mode == "RGBA" {
                    Checkerboard().frame(width: fitted.width, height: fitted.height)
                }
                if let nsImage {
                    Image(nsImage: nsImage)
                        .resizable()
                        .interpolation(.high)
                        .frame(width: fitted.width, height: fitted.height)
                } else {
                    Thumbnail(path: image.path, size: 400).frame(width: fitted.width, height: fitted.height)
                }
            }
            .shadow(color: .black.opacity(0.2), radius: 10, y: 4)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .draggable(image.url)
            .contextMenu { ImageActions(image: image) }
        }
        .task(id: image.path) {
            nsImage = nil
            let path = image.path
            nsImage = await Task.detached { NSImage(contentsOfFile: path) }.value
        }
    }

    private func fittedSize(in box: CGSize) -> CGSize {
        let scale = min(box.width / CGFloat(image.width), box.height / CGFloat(image.height), 1)
        return CGSize(width: max(1, CGFloat(image.width) * scale), height: max(1, CGFloat(image.height) * scale))
    }
}

// MARK: - History

private struct HistoryStrip: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ScrollView(.horizontal) {
            LazyHStack(spacing: 8) {
                ForEach(model.activeJobs) { job in
                    RoundedRectangle(cornerRadius: 8)
                        .fill(.quaternary)
                        .frame(width: 84, height: 84)
                        .overlay {
                            if job.status == "running" {
                                ProgressView(value: job.fraction).progressViewStyle(.circular).controlSize(.small)
                            } else {
                                Image(systemName: "clock").foregroundStyle(.secondary)
                            }
                        }
                        .overlay(RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(model.showsJobProgress ? Color.accentColor : .clear, lineWidth: 2.5))
                        .help(job.params.prompt)
                        .onTapGesture { model.showsJobProgress = true }
                }
                ForEach(model.history) { image in
                    Thumbnail(path: image.path, size: 84)
                        .frame(width: 84, height: 84)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay(RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(model.selectedImage == image && !model.showsJobProgress ? Color.accentColor : .clear, lineWidth: 2.5))
                        .overlay(alignment: .bottomTrailing) {
                            if image.isFromMCP {
                                Image(systemName: "point.3.connected.trianglepath.dotted")
                                    .font(.caption2)
                                    .padding(3)
                                    .background(.black.opacity(0.55), in: Circle())
                                    .foregroundStyle(.white)
                                    .padding(4)
                            }
                        }
                        .onTapGesture { model.selectedImage = image }
                        .draggable(image.url)
                        .contextMenu { ImageActions(image: image) }
                        .help(image.prompt)
                }
                if model.history.isEmpty && model.activeJobs.isEmpty {
                    Text("Your images appear here").foregroundStyle(.tertiary).padding(.leading, 8)
                }
            }
            .padding(10)
        }
        .frame(height: 104)
        .background(.background)
    }
}

struct ImageActions: View {
    @Environment(AppModel.self) private var model
    let image: GeneratedImage

    var body: some View {
        Button("Use Settings") { model.reuseSettings(of: image) }
        Button("Use as Reference") { model.addReference(image.url) }
        Divider()
        Button("Copy Image") { Clipboard.copyImage(at: image.url) }
        Button("Copy Path") { Clipboard.copy(image.path) }
        Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([image.url]) }
        Button("Open in Preview") { NSWorkspace.shared.open(image.url) }
        Divider()
        Button("Delete", role: .destructive) { model.delete(image) }
    }
}
