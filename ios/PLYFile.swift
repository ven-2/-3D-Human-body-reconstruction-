//
//  PLYFile.swift
//  ProjectCloudExample
//
//  Created by Venkata Ashokkumar on 19/01/2026.
//
import Foundation
import SwiftUI
import UniformTypeIdentifiers

struct PLYFile: Transferable {

    let pointCloud: PointCloud

    enum ExportError: LocalizedError {
        case cannotExport
    }

    func export() async throws -> Data {
        let verts = await pointCloud.snapshotVertices()

        var ply = """
        ply
        format ascii 1.0
        element vertex \(verts.count)
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        """

        for v in verts {
            let x = v.position.x
            let y = v.position.y
            let z = v.position.z

            
            let r = UInt8(max(0, min(255, Int(v.color.x * 255))))
            let g = UInt8(max(0, min(255, Int(v.color.y * 255))))
            let b = UInt8(max(0, min(255, Int(v.color.z * 255))))
            let a = UInt8(max(0, min(255, Int(v.color.w * 255))))

            ply += "\n\(x) \(y) \(z) \(r) \(g) \(b) \(a)"
        }

        guard let data = ply.data(using: .ascii) else {
            throw ExportError.cannotExport
        }
        return data
    }

    static var transferRepresentation: some TransferRepresentation {
        DataRepresentation(exportedContentType: .data) { file in
            try await file.export()
        }
        .suggestedFileName("exported.ply")
    }
}
