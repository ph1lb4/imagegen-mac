import Foundation

// MARK: - Wire types (mirror backend/imagegen/app.py)

struct DownloadState: Codable, Equatable {
    var status: String
    var bytesDone: Int64
    var bytesTotal: Int64
    var currentFile: String?
    var speedBps: Double
    var error: String?
    var filesDone: Int
    var filesTotal: Int

    var fraction: Double { bytesTotal > 0 ? Double(bytesDone) / Double(bytesTotal) : 0 }
    var isActive: Bool { status == "downloading" }
}

struct ModelInfo: Codable, Identifiable, Equatable {
    var id: String
    var name: String
    var repo: String
    var url: String
    var description: String
    var license: String
    var sizeGb: Double
    var recommendedMemoryGb: Int
    var capabilities: [String]
    var defaultSteps: Int
    var fixedSteps: Bool?
    var maxReferenceImages: Int
    var aspectRatios: [String]
    var installed: Bool
    var localPath: String
    var download: DownloadState?
}

struct EngineState: Codable, Equatable {
    var modelId: String?
    var status: String
    var message: String
    var device: String?
    var gpuMemoryGb: Double?
    var gpuMemoryLimitGb: Double?
    var maxResolution: Int?

    var isReady: Bool { status == "ready" }
    var isLoading: Bool { status == "loading" }
}

struct JobParams: Codable, Equatable {
    var prompt: String
    var negativePrompt: String?
    var aspectRatio: String
    var resolution: Int
    var width: Int?
    var height: Int?
    var steps: Int?
    var seed: Int?
    var cfgScale: Double
    var transparent: Bool
    var referenceImages: [String]
}

struct GeneratedImage: Codable, Identifiable, Hashable {
    var name: String
    var path: String
    var width: Int
    var height: Int
    var mode: String?
    var modelId: String
    var source: String
    var jobId: String
    var createdAt: Double
    var durationS: Double
    var prompt: String
    var negativePrompt: String?
    var aspectRatio: String?
    var resolution: Int?
    var steps: Int?
    var seed: Int?
    var cfgScale: Double?
    var transparent: Bool?
    var referenceImages: [String]?

    var id: String { name }
    var url: URL { URL(fileURLWithPath: path) }
    var createdDate: Date { Date(timeIntervalSince1970: createdAt) }
    var isFromMCP: Bool { source == "mcp" }
}

struct Job: Codable, Identifiable, Equatable {
    var id: String
    var params: JobParams
    var modelId: String
    var source: String
    var status: String
    var step: Int
    var totalSteps: Int
    var stage: String
    var createdAt: Double
    var startedAt: Double?
    var finishedAt: Double?
    var image: GeneratedImage?
    var error: String?

    var isActive: Bool { status == "queued" || status == "running" }
    var fraction: Double { totalSteps > 0 ? Double(step) / Double(totalSteps) : 0 }
}

struct StatusResponse: Codable {
    var engine: EngineState
    var jobs: [Job]
    var mcpUrl: String
    var outputDir: String
    var freeDiskGb: Double
}

struct GenerateRequest: Codable {
    var prompt: String
    var negativePrompt: String?
    var aspectRatio: String
    var resolution: Int
    var steps: Int?
    var seed: Int?
    var cfgScale: Double
    var transparent: Bool
    var referenceImages: [String]
}

struct APIError: LocalizedError {
    var message: String
    var errorDescription: String? { message }
}

// MARK: - Client

struct APIClient {
    var baseURL: URL

    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        return e
    }()

    private static let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 10
        return URLSession(configuration: config)
    }()

    func health() async -> Bool {
        var request = URLRequest(url: baseURL.appending(path: "api/health"))
        request.timeoutInterval = 2
        guard let (_, response) = try? await Self.session.data(for: request) else { return false }
        return (response as? HTTPURLResponse)?.statusCode == 200
    }

    func status() async throws -> StatusResponse { try await get("api/status") }
    func models() async throws -> [ModelInfo] { try await get("api/models") }
    func history() async throws -> [GeneratedImage] { try await get("api/history") }

    @discardableResult
    func download(_ id: String) async throws -> ModelInfo { try await send("POST", "api/models/\(id)/download") }
    func cancelDownload(_ id: String) async throws { let _: [String: Bool] = try await send("POST", "api/models/\(id)/cancel-download") }
    @discardableResult
    func deleteModel(_ id: String) async throws -> ModelInfo { try await send("DELETE", "api/models/\(id)") }
    @discardableResult
    func load(_ id: String) async throws -> EngineState { try await send("POST", "api/models/\(id)/load") }
    @discardableResult
    func unload() async throws -> EngineState { try await send("POST", "api/unload") }
    func generate(_ req: GenerateRequest) async throws -> Job { try await send("POST", "api/generate", body: req) }
    func cancelJob(_ id: String) async throws { let _: [String: Bool] = try await send("POST", "api/jobs/\(id)/cancel") }
    func deleteImage(_ name: String) async throws { let _: [String: Bool] = try await send("DELETE", "api/images/\(name)") }

    private func get<T: Decodable>(_ path: String) async throws -> T {
        try await send("GET", path)
    }

    private func send<T: Decodable>(_ method: String, _ path: String, body: (some Encodable)? = nil as String?) async throws -> T {
        var request = URLRequest(url: baseURL.appending(path: path))
        request.httpMethod = method
        if let body {
            request.httpBody = try Self.encoder.encode(body)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let (data, response) = try await Self.session.data(for: request)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            let detail = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["detail"] as? String
            throw APIError(message: detail ?? "Request failed (\(code))")
        }
        return try Self.decoder.decode(T.self, from: data)
    }
}
