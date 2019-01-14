//
//  ViewController.swift
//  CoreMLAndVisionDemo
//
//  Created by LYL on 2019/1/8.
//  Copyright © 2019 LYL. All rights reserved.
//

import UIKit


import CoreML
import Vision

class ViewController: UIViewController {

    @IBOutlet weak var imgView: UIImageView!
    @IBOutlet weak var photoLibrary: UIButton!
    @IBOutlet weak var indicator: UIActivityIndicatorView!
    
    lazy var classificationRequest: VNCoreMLRequest = {
        guard let model = try? VNCoreMLModel(for: GoogLeNetPlaces().model) else {
            fatalError("load GoogLeNetPlaces model error")
        }
        
        let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] (request, error) in
            self?.processClassifications(for: request, error: error)
        })
        request.imageCropAndScaleOption = .centerCrop
        return request
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        imgView.layer.borderWidth = 1.0
        imgView.layer.borderColor = UIColor.orange.cgColor
    }
    
    @IBAction func selectPhoto(_ sender: UIButton) {
        let imgPicker = UIImagePickerController()
        imgPicker.delegate = self
        imgPicker.sourceType = .photoLibrary
        present(imgPicker, animated: true, completion: nil)
    }
}

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let image = info[.originalImage] as? UIImage {
            imgView.image = image
//            analysisImageWithoutVision(image: image)
            analysisImageWithVision(image: image)
        }
        picker.dismiss(animated: true, completion: nil)
    }
}

extension ViewController {
    // 展示分析结果
    func showAnalysisResultOnMainQueue(with message: String) {
        DispatchQueue.main.async {
            let alert = UIAlertController(title: "Completed", message: message, preferredStyle: .alert)
            let cancelAct = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
            alert.addAction(cancelAct)
            self.present(alert, animated: true) {
                self.indicator.stopAnimating()
                self.view.isUserInteractionEnabled = true
            }
        }
    }
    
    func createPixelBufferPool(_ width: Int32, _ height: Int32, _ pixelFormat: FourCharCode, _ maxBufferCount: Int32) -> CVPixelBufferPool? {
        var outputPool: CVPixelBufferPool? = nil
        let sourcePixelBufferOptions: NSDictionary = [kCVPixelBufferPixelFormatTypeKey: pixelFormat,
                                                                kCVPixelBufferWidthKey: width,
                                                                kCVPixelBufferHeightKey: height,
                                                                kCVPixelFormatOpenGLESCompatibility: true,
                                                                kCVPixelBufferIOSurfacePropertiesKey: NSDictionary()]
        let pixelBufferPoolOptions: NSDictionary = [kCVPixelBufferPoolMinimumBufferCountKey: maxBufferCount]
        CVPixelBufferPoolCreate(kCFAllocatorDefault,
                                pixelBufferPoolOptions,
                                sourcePixelBufferOptions,
                                &outputPool)
        return outputPool
    }
    
    // 将一个UIImage类型的图片对象，转换成一个CVPixelBuffer类型的对象
    func CreatePixelBufferFromImage(_ image: UIImage) -> CVPixelBuffer? {
        let size = image.size
        var pxbuffer : CVPixelBuffer?
        let pixelBufferPool = createPixelBufferPool(Int32(size.width),
                                                    Int32(size.height),
                                                    FourCharCode(kCVPixelFormatType_32BGRA),
                                                    2056)
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pixelBufferPool!, &pxbuffer)
        guard (status == kCVReturnSuccess) else{
            return nil
        }
        CVPixelBufferLockBaseAddress(pxbuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pxdata = CVPixelBufferGetBaseAddress(pxbuffer!)
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pxdata,
                                width: Int(size.width),
                                height: Int(size.height),
                                bitsPerComponent: 8,
                                bytesPerRow: CVPixelBufferGetBytesPerRow(pxbuffer!),
                                space: rgbColorSpace,
                                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue)
        context?.translateBy(x: 0, y: image.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)
        UIGraphicsPushContext(context!)
        image.draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pxbuffer!, CVPixelBufferLockFlags(rawValue: 0))
        return pxbuffer
    }
}

extension ViewController {
    // 使用纯CoreML进行图片分析
    func analysisImageWithoutVision(image: UIImage) {
        // 关闭交互
        indicator.startAnimating()
        view.isUserInteractionEnabled = false
        //---------------- 1 ----------------
        DispatchQueue.global(qos: .userInteractive).async {
            //---------------- 2 ----------------
            let imageWidth: CGFloat = 224.0
            let imageHeight: CGFloat = 224.0
            UIGraphicsBeginImageContext(CGSize(width:imageWidth, height:imageHeight))
            image.draw(in:CGRect(x:0, y:0, width:imageHeight, height:imageHeight))
            let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
            UIGraphicsEndImageContext()
            guard let newImage = resizedImage else {
                fatalError("resized Image fail")
            }
            //---------------- 3 ----------------
            guard let pixelBuffer = self.CreatePixelBufferFromImage(newImage) else {
                fatalError("convert PixelBuffer fail")
            }
            // ---------------- 4 ----------------
            guard let output = try? GoogLeNetPlaces().prediction(sceneImage: pixelBuffer) else {
                fatalError("predict fail")
            }
            // ---------------- 5 ----------------
            let result = "\(output.sceneLabel)(\(Int(output.sceneLabelProbs[output.sceneLabel]! * 100))%)"
            // ---------------- 6 ----------------
            self.showAnalysisResultOnMainQueue(with: result)
        }
    }
    
    // 使用CoreML+Vision方法来分析
    func analysisImageWithVision(image: UIImage) {
        indicator.startAnimating()
        view.isUserInteractionEnabled = false
        
        guard let ciImage = CIImage(image: image) else {
            fatalError("convert CIImage error")
        }

        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
            do {
                try handler.perform([self.classificationRequest])
            } catch {
                print("Failed to perform classification.\n\(error.localizedDescription)")
            }
        }
    }
    
    func processClassifications(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            guard let results = request.results else {
                let result = "Unable to classify image.\n\(error!.localizedDescription)"
                self.showAnalysisResultOnMainQueue(with: result)
                return
            }
            let classifications = results as! [VNClassificationObservation]
            if classifications.isEmpty {
                let result = "Nothing recognized."
                self.showAnalysisResultOnMainQueue(with: result)
            } else {
                let topClassifications = classifications.prefix(2)
                let descriptions = topClassifications.map { classification in
                    return String(format: "  (%.2f) %@", classification.confidence, classification.identifier)
                }
                let result = "Classification:\n" + descriptions.joined(separator: "\n")
                self.showAnalysisResultOnMainQueue(with: result)
            }
        }
    }
}
