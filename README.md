The project enables Julia and Java programmers to experiment with Deep Learning comparatively.

To build use:

mvn clean package

Then:
java -jar julia-dl4j-ide-1.0-SNAPSHOT.jar

To execute a DL4j model update the code of the method:


    private void executeDL4J() {
        consoleOutput.append("\n>>> Initializing DeepLearning4j Network...\n");
        try {
            // Define a simple Neural Network
            var conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .activation(Activation.SOFTMAX).nIn(3).nOut(2).build())
                .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            // Perform a dummy inference
            var input = Nd4j.rand(1, 4);
            var output = net.output(input);

            consoleOutput.append("DL4J Success! Input: " + input + "\n");
            consoleOutput.append("Prediction: " + output + "\n");
        } catch (Exception ex) {
            consoleOutput.append("DL4J Error: " + ex.getMessage() + "\n");
        }
    }

    within the JuliaIDE.java file.

    To run Julia code type in the editor and execute with the corresponding button.

    
Java DL4j code can be executed and with the JShell based option.




