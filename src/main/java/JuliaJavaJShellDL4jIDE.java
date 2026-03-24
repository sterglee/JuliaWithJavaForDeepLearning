import org.fife.ui.rsyntaxtextarea.RSyntaxTextArea;
import org.fife.ui.rsyntaxtextarea.SyntaxConstants;
import org.fife.ui.rtextarea.RTextScrollPane;
import org.fife.ui.autocomplete.*;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import jdk.jshell.JShell;
import jdk.jshell.Snippet;
import jdk.jshell.SnippetEvent;
import jdk.jshell.SourceCodeAnalysis;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.nio.file.Files;
import java.util.List;

public class JuliaJavaJShellDL4jIDE extends JFrame {
    private RSyntaxTextArea codeEditor;
    private JTextArea consoleOutput;
    private JButton runJuliaBtn, runDL4JBtn, runJShellBtn, clearBtn, stopBtn;
    private JShell jshell;
    private Process currentProcess; // Track external Julia process

    public JuliaJavaJShellDL4jIDE() {
        setTitle("Hybrid Julia & DL4J IDE");
        setSize(1100, 850);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        initJShell();

        codeEditor = new RSyntaxTextArea(25, 80);
        codeEditor.setSyntaxEditingStyle(SyntaxConstants.SYNTAX_STYLE_JAVA);
        codeEditor.setTabSize(4);
        codeEditor.setAnimateBracketMatching(true);
        
        DefaultCompletionProvider provider = new DefaultCompletionProvider();
        provider.addCompletion(new BasicCompletion(provider, "import org.nd4j.linalg.factory.Nd4j;"));
        new AutoCompletion(provider).install(codeEditor);

        consoleOutput = new JTextArea(15, 60);
        consoleOutput.setEditable(false);
        consoleOutput.setBackground(new Color(20, 20, 25));
        consoleOutput.setForeground(new Color(50, 255, 150));
        consoleOutput.setFont(new Font("Monospaced", Font.PLAIN, 13));

        runJuliaBtn = new JButton("Run Julia");
        runDL4JBtn = new JButton("Run Internal DL4J");
        runJShellBtn = new JButton("Execute JShell");
        clearBtn = new JButton("Clear");
        stopBtn = new JButton("Stop Process");
        stopBtn.setForeground(Color.RED);

        JPanel btnPanel = new JPanel();
        btnPanel.add(runJuliaBtn); btnPanel.add(runDL4JBtn);
        btnPanel.add(runJShellBtn); btnPanel.add(stopBtn); btnPanel.add(clearBtn);

        JSplitPane split = new JSplitPane(JSplitPane.VERTICAL_SPLIT, 
                new RTextScrollPane(codeEditor), new JScrollPane(consoleOutput));
        split.setDividerLocation(500);

        add(split, BorderLayout.CENTER);
        add(btnPanel, BorderLayout.SOUTH);

        runJuliaBtn.addActionListener(e -> executeJulia());
        runDL4JBtn.addActionListener(e -> executeDL4J());
        runJShellBtn.addActionListener(e -> executeJShell());
        clearBtn.addActionListener(e -> consoleOutput.setText(""));
        stopBtn.addActionListener(e -> stopCurrentProcess());
    }

    private void initJShell() {
        String verStr = String.valueOf(Runtime.version().feature());
        try {
            this.jshell = JShell.builder()
                    .remoteVMOptions("--add-modules", "java.desktop")
                    .compilerOptions("--release", verStr, "--add-modules", "java.desktop")
                    .build();
            
            // Pre-load common imports for convenience
            jshell.eval("import org.nd4j.linalg.factory.Nd4j;");
            jshell.eval("import org.nd4j.linalg.api.ndarray.INDArray;");
            
            System.out.println("System: JShell Ready (Java " + verStr + ").");
        } catch (Exception ex) {
            this.jshell = JShell.create();
            System.out.println("System: JShell started in standard mode.");
        }

        // Add local libraries to classpath
        File libDir = new File("lib");
        if (libDir.exists() && libDir.isDirectory()) {
            File[] jars = libDir.listFiles((dir, name) -> name.endsWith(".jar"));
            if (jars != null) {
                for (File jar : jars) {
                    jshell.addToClasspath(jar.getAbsolutePath());
                }
            }
        }
    }

    private void executeJShell() {
        if (jshell == null) return;
        String fullContent = codeEditor.getText().trim();
        if (fullContent.isEmpty()) return;

        System.out.println("\n>>> JShell Execution:");
        
        String remaining = fullContent;
        while (!remaining.isEmpty()) {
            SourceCodeAnalysis.CompletionInfo info = jshell.sourceCodeAnalysis().analyzeCompletion(remaining);
            String snippetSource = info.source();
            
            if (snippetSource.isEmpty()) break;

            List<SnippetEvent> events = jshell.eval(snippetSource);
            for (SnippetEvent e : events) {
                if (e.status() == Snippet.Status.VALID) {
                    if (e.value() != null && !e.value().isEmpty()) {
                        System.out.println("Out: " + e.value());
                    }
                } else {
                    jshell.diagnostics(e.snippet()).forEach(d -> 
                        System.out.println("Compile Error: " + d.getMessage(null))
                    );
                }
                if (e.exception() != null) {
                    System.out.println("Runtime Error: " + e.exception().getMessage());
                }
            }
            remaining = info.remaining().trim();
        }
    }

    private void executeDL4J() {
        System.out.println("\n>>> Internal: Running DL4J Demo...");
        try {
            // Check if ND4J is actually on the classpath
            Nd4j.getBackend();
            var conf = new NeuralNetConfiguration.Builder()
                .seed(123).list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .activation(Activation.SOFTMAX).nIn(3).nOut(2).build())
                .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            System.out.println("DL4J Success! Weights Initialized.");
            System.out.println("Sample Output: " + net.output(Nd4j.rand(1, 4)));
        } catch (NoClassDefFoundError | Exception ex) {
            System.out.println("Internal DL4J Error: Library not found or " + ex.getMessage());
        }
    }

    private void executeJulia() {
        runJuliaBtn.setEnabled(false);
        System.out.println("\n>>> Julia: Running Script...");
        
        new SwingWorker<Void, String>() {
            @Override
            protected Void doInBackground() throws Exception {
                File tempJl = File.createTempFile("ide_script", ".jl");
                Files.writeString(tempJl.toPath(), codeEditor.getText());
                
                ProcessBuilder pb = new ProcessBuilder("julia", tempJl.getAbsolutePath());
                pb.redirectErrorStream(true);
                currentProcess = pb.start();

                try (BufferedReader reader = new BufferedReader(new InputStreamReader(currentProcess.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        publish(line);
                    }
                }
                currentProcess.waitFor();
                return null;
            }

            @Override 
            protected void process(List<String> chunks) { 
                for (String s : chunks) System.out.println(s); 
            }

            @Override 
            protected void done() { 
                runJuliaBtn.setEnabled(true);
                currentProcess = null;
                System.out.println(">>> Julia: Finished.");
            }
        }.execute();
    }

/**
 * Appends text to the console JTextArea in a thread-safe manner.
 * Ensures the scroll pane automatically follows the latest output.
 */
public void appendConsole(String text) {
    // Swing is not thread-safe. Since JShell and Julia run on background 
    // threads, we must wrap UI updates in invokeLater.
    SwingUtilities.invokeLater(() -> {
        // 1. Append the text and a newline
        consoleOutput.append(text + "\n");

        // 2. Force the caret to the end of the document to trigger auto-scroll
        consoleOutput.setCaretPosition(consoleOutput.getDocument().getLength());
        
        // 3. Optional: Limit console size to prevent memory lag over long sessions
        if (consoleOutput.getLineCount() > 2000) {
            try {
                int end = consoleOutput.getLineEndOffset(500); // Remove first 500 lines
                consoleOutput.replaceRange("", 0, end);
            } catch (Exception ignored) {
                // Silently fail if clearing fails
            }
        }
    });
}

    private void stopCurrentProcess() {
        if (currentProcess != null && currentProcess.isAlive()) {
            currentProcess.destroyForcibly();
            System.out.println("\n[System] External process terminated.");
        } else {
            System.out.println("\n[System] No external process currently running.");
        }
    }

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {}
        
        SwingUtilities.invokeLater(() -> new JuliaIDE().setVisible(true));
    }
}

