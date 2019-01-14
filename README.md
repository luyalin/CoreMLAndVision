# CoreMLAndVision
机器学习是一种人工智能，计算机在没有明确编程的情况下进行“学习”。机器学习工具并不是编码一个算法，而是通过在大量数据中查找模式，使计算机能够开发和优化算法。

####发展历程
Apple最早在 iOS 5 中引入了 NSLinguisticTagger 来分析自然语言，将文本分段为段落、句子或单词，并标记有关信息，如词性（动词、名词、形容词）。

在 iOS 8 中，引入了 Metal，提供了对设备 GPU 底层访问接口，为机器学习提供强大的计算能力。

2016年，Apple 在其 Accelerate 框架中添加了 BNNS，使开发人员能够为已经训练过的神经网络提供非常高的性能推断（而不是训练）。

虽然机器学习和计算机视觉在 iOS 上早已有了系统级的支持，但直到2017年，Apple 推出了 Core ML 和 Vision之后，由于其设计合理且容易上手的API，使得使用门槛大大降低。

![层级结构](https://upload-images.jianshu.io/upload_images/2517018-f3f6f89f2bff7d38.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

####Core ML简介
Core ML能够将数据经过预处理后输入MLMODEL文件，输出为模型的预测结果。使用 Core ML 只需要很少的代码就可以构建起一个机器学习的应用。只需关注代码即可，无需关注模型的定义、网络的构成。Core ML 使得开发者能够将各种各样的机器学习模型集成到应用程序中。它除了支持超过 30 层类型的广泛深度学习，还支持如树集成、SVMs 和广义线性模型等标准模型。因为苹果制定了自己的模型文件格式，使得Core ML支持苹果生态下多个平台（macOS，iOS，watchOS，TVOS）。

![MLMODEL](https://upload-images.jianshu.io/upload_images/2517018-ffb0e5ceeb5a6842.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

此外，苹果还提供了一个 Python 工具，可以将业内一些常用的机器学习框架导出的 Model 转成 MLMODEL 文件，如DNN,RNN,CNN,SVM,Tree ensembles,Generalized linear models,Pipeline models等。

![model转换](https://upload-images.jianshu.io/upload_images/2517018-6b7f5de59657c88a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

####Vision简介
Vision是一个高性能的图片分析库，它能识别在图片和视频中的人脸、特征、场景分类等。它包含的分类有Face Detection and Recognition(人脸检测识别)、Machine Learning Image Analysis(机器学习图片分析)、Barcode Detection(条形码检测)、Text Detection(文本检测)等。Vision 操作流水线分为两类：分析图片和跟踪队列。可以使用图片检测出的物体或矩形结果（Observation）来作为跟踪队列请求（Request）的参数。

![分析图片](https://upload-images.jianshu.io/upload_images/2517018-14d17a63f4b75b5b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![跟踪队列](https://upload-images.jianshu.io/upload_images/2517018-f683d4aece61e765.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

####使用方法

下面就以一个图片检测识别的例子展开讲解使用方法。
1、获取模型。从官网下载（https://developer.apple.com/machine-learning/build-run-models/）或使用Core ML Tools转化第三方框架生成的模型，也可以使用2018年推出的原生Create ML 框架来直接从数据生成 Core ML 模型。这里直接下载官网提供的Places205-GoogLeNet为例子。

![GoogLeNet](https://upload-images.jianshu.io/upload_images/2517018-f621c8c8a6b22d27.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

2、导入并生成接口。获取到模型后直接拖拽导入Xcode，Xcode会自动生成相应的机器学习模型接口。无需任何手动或其他操作，十分方便友好。

![Model Class](https://upload-images.jianshu.io/upload_images/2517018-fc4b5520b38d8ee7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上图中Model Class中可以看到，Xcode会自动生成Swift model类，点击箭头就能查看模型接口。

![API](https://upload-images.jianshu.io/upload_images/2517018-ad21eed0dfb4330c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

3、使用编程接口。根据生成的 API 进行编程。首先需要在控制器中导入两个框架：
>import CoreML
>import Vision

接着，通过懒加载创建一个请求，并在回调中处理结果。VNCoreMLModel是与 Vision 请求一起使用的 Core ML 模型的容器。Vision的工作流程就是：
1、创建模型
2、创建一个或多个请求
3、创建并运行请求处理程序。
~~~swift
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
~~~
然后，使用Vision框架请求处理程序。将一个表示图像的参数传递给VNImageRequestHandler，然后调用它的perform方法并传递一组请求对象，来运行处理程序。
~~~swift
DispatchQueue.global(qos: .userInitiated).async {
let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
do {
try handler.perform([self.classificationRequest])
} catch {
print("Failed to perform classification.\n\(error.localizedDescription)")
} 
}
~~~
当请求完成后，就会执行之前回调中的处理结果的逻辑，即processClassifications方法。在这个方法中，通过参数resquest.results会得到一组VNClassificationObservation对象，这个对象有两个属性是confidence和identifier，分别表示信任值和类别名，信任值是范围0~1之间。
~~~swift
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
~~~
因为results数组是按照信任值降序排列的，所以代码中直接取前两个结果展示出来。
demo给出的两张图片检测结果：
![demo1](https://upload-images.jianshu.io/upload_images/2517018-4b97b9f5c3e4ada7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![demo2](https://upload-images.jianshu.io/upload_images/2517018-95b6e068af345df6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

####总结
苹果通过把复杂的事情简单化，降低使用门槛，并且对一些高频场景进行了封装，比如人脸、条形码、文字等，很容易激发开发者的热情。另外，因为移动端训练模型意义较小，它选择另辟蹊径，将模型训练交给服务端去做，使得它能够更大发挥硬件上性能的优势。

