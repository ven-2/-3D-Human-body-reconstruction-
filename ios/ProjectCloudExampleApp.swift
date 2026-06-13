//
//  ProjectCloudExampleApp.swift
//  ProjectCloudExample
//
//  Created by Venkata Ashokkumar on 18/01/2026.
import SwiftUI
import UIKit

struct UIViewWrapper<V: UIView>: UIViewRepresentable {
    let view: V
    func makeUIView(context: Context) -> V { view }
    func updateUIView(_ uiView: V, context: Context) {}
}

@main
struct ProjectCloudExampleApp: App {

    @State private var arManager = ARManager()

    var body: some Scene {
        WindowGroup {
            ZStack(alignment: .bottom) {
                UIViewWrapper(view: arManager.sceneView)
                    .ignoresSafeArea()

                HStack(spacing: 30) {
                    Button {
                        arManager.toggleCapture()
                    } label: {
                        Image(systemName: arManager.isCapturing
                              ? "stop.circle.fill"
                              : "play.circle.fill")
                    }

                    ShareLink(
                        item: PLYFile(pointCloud: arManager.pointCloud),
                        preview: SharePreview("exported.ply")
                    ) {
                        Image(systemName: "square.and.arrow.up.circle.fill")
                    }

                    if let url = arManager.resultFileURL {
                        ShareLink(item: url, preview: SharePreview("result.ply")) {
                            Image(systemName: "tray.and.arrow.down.fill")
                        }
                    }
                }
                .foregroundStyle(.black, .white)
                .font(.system(size: 50))
                .padding(25)
            }
        }
    }
}
