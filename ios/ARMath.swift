//
//  ARMath.swift
//  ProjectCloudExample
//
//  Created by Venkata Ashokkumar on 18/01/2026.
//
import simd
import UIKit

func makeRotateToARCameraMatrix(orientation: UIInterfaceOrientation) -> matrix_float4x4 {
    let flipYZ = matrix_float4x4(
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    )

    let rotationAngle: Float = switch orientation {
    case .landscapeLeft: .pi
    case .portrait: .pi / 2
    case .portraitUpsideDown: -.pi / 2
    default: 0
    }

    let q = simd_quaternion(rotationAngle, simd_float3(0, 0, 1))
    let rotationMatrix = matrix_float4x4(q)

    return flipYZ * rotationMatrix
}

