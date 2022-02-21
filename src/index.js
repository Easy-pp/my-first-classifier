const IMAGE_SIZE = 244;

const loadImg = (src) => {
    return new Promise((resolve) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.src = src;
        img.width = IMAGE_SIZE;
        img.height = IMAGE_SIZE;
        img.onload = () => resolve(img);
    });
};

(async () => {
    let net;
    const imgContainer = document.getElementById('container');
    const classifier = knnClassifier.create();
    async function app() {
        console.log('Loading mobilenet..');

        // Load the model.
        net = await mobilenet.load();
        console.log('Successfully loaded model');
        // Reads an image from the webcam and associates it with a specific class
        // index.
        const addExample = async (imgEl, classId) => {
            const img = tf.browser.fromPixels(imgEl);
            // Get the intermediate activation of MobileNet 'conv_preds' and pass that
            // to the KNN classifier.
            const activation = net.infer(img, true);

            // Pass the intermediate activation to the classifier.
            classifier.addExample(activation, classId);

            // Dispose the tensor to release the memory.
            // img.dispose();
        };
        const loadExample = async (mobileId, count = 0) => {
            const imgs = [];
            // load mobile example
            for (let i = 1; i <= count; i++) {
                const src = `http://127.0.0.1:8080/${mobileId}/${mobileId}-${i}.png`;
                const img = loadImg(src);
                imgs.push(img);
            }
            const mobileImgs = await Promise.all(imgs);
            mobileImgs.forEach(img => addExample(img, mobileId))
        }
        await loadExample('close', 62);
        await loadExample('phone', 49);
        await loadExample('mobile', 25);

        console.log('add example end', classifier)
        console.log(classifier.getClassifierDataset())
        const consoleContainer = document.getElementById('console');
        // 点击预测图片
        document.querySelectorAll('.imgs').forEach(item => {
            item.addEventListener('click', async () => {
                const img = tf.browser.fromPixels(item);
                // Get the activation from mobilenet from the webcam.
                const activation = net.infer(img, 'conv_preds');
                // Get the most likely class and confidence from the classifier module.
                const result = await classifier.predictClass(activation, 2);
                console.log(result);
                consoleContainer.innerHTML = result.label;
            })
        })
    }

    app();
})()