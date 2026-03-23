import org.fife.ui.rsyntaxtextarea.RSyntaxTextArea;
import org.fife.ui.rtextarea.RTextScrollPane;
import org.fife.ui.autocomplete.*;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.nio.file.Files;

public class JuliaIDE extends JFrame {
    private RSyntaxTextArea codeEditor;
    private JTextArea consoleOutput;
    private JButton runJuliaBtn, runDL4JBtn;

    public JuliaIDE() {
        setTitle("Hybrid Julia & DeepLearning4j IDE");
        setSize(1000, 750);
        setDefaultCloseOperation(EXIT_ON_CLOSE);

        // 1. Editor Setup
        codeEditor = new RSyntaxTextArea(20, 60);
        codeEditor.setSyntaxEditingStyle("text/julia");
        
        // 2. Auto-completion Setup
        DefaultCompletionProvider provider = new DefaultCompletionProvider();
        provider.addCompletion(new BasicCompletion(provider, "function"));
        provider.addCompletion(new BasicCompletion(provider, "println"));
        new AutoCompletion(provider).install(codeEditor);

        // 3. Console & Buttons
        consoleOutput = new JTextArea(12, 60);
        consoleOutput.setEditable(false);
        consoleOutput.setBackground(new Color(30, 30, 30));
        consoleOutput.setForeground(Color.GREEN);
        consoleOutput.setFont(new Font("Monospaced", Font.PLAIN, 13));

        runJuliaBtn = new JButton("Run Julia Script");
        runDL4JBtn = new JButton("Run Java DL4J Model");

        // Layout
        JPanel btnPanel = new JPanel();
        btnPanel.add(runJuliaBtn);
        btnPanel.add(runDL4JBtn);

        JSplitPane split = new JSplitPane(JSplitPane.VERTICAL_SPLIT, 
                new RTextScrollPane(codeEditor), new JScrollPane(consoleOutput));
        split.setDividerLocation(450);

        add(split, BorderLayout.CENTER);
        add(btnPanel, BorderLayout.SOUTH);

        // Action Listeners
        runJuliaBtn.addActionListener(e -> executeJulia());
        runDL4JBtn.addActionListener(e -> executeDL4J());
    }

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

    private void executeJulia() {
        runJuliaBtn.setEnabled(false);
        consoleOutput.append("\n>>> Running Julia...\n");
        new SwingWorker<Void, String>() {
            @Override
            protected Void doInBackground() throws Exception {
                File f = File.createTempFile("exec", ".jl");
                Files.writeString(f.toPath(), codeEditor.getText());
                Process p = new ProcessBuilder("julia", f.getAbsolutePath()).redirectErrorStream(true).start();
                try (var r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                    String l; while ((l = r.readLine()) != null) publish(l);
                }
                p.waitFor();
                return null;
            }
            @Override protected void process(java.util.List<String> c) { for (String s : c) consoleOutput.append(s + "\n"); }
            @Override protected void done() { runJuliaBtn.setEnabled(true); }
        }.execute();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new JuliaIDE().setVisible(true));
    }
}

