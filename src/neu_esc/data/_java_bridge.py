"""Java source embedded so wheels need no extra package-data declarations."""

BRIDGE_SOURCE = r'''
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.ArrayList;
import vn.pipeline.Annotation;
import vn.pipeline.Sentence;
import vn.pipeline.VnCoreNLP;

/** One isolated VnCoreNLP JVM, one base64 UTF-8 request/response per source row. */
class WsegBridge {
    public static void main(String[] args) throws Exception {
        // Upstream derives jarDir from URL.getPath(), leaving %20 in Windows paths.
        // Set its public resource locator to the actual cwd before loading models.
        // This changes file lookup only; the locked algorithm/models are untouched.
        vn.pipeline.Utils.jarDir = new File(".").getCanonicalPath();
        VnCoreNLP pipeline = new VnCoreNLP(new String[] {"wseg"});
        BufferedReader input = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        PrintWriter output = new PrintWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8), true);
        output.println("NEU_READY");
        String line;
        while ((line = input.readLine()) != null) {
            String text = new String(Base64.getDecoder().decode(line), StandardCharsets.UTF_8);
            Annotation annotation = new Annotation(text);
            pipeline.annotate(annotation);
            ArrayList<String> sentences = new ArrayList<>();
            for (Sentence sentence : annotation.getSentences()) {
                sentences.add(sentence.getWordSegmentedSentence());
            }
            String segmented = String.join(" ", sentences);
            output.println("NEU_WSEG\t" + Base64.getEncoder().encodeToString(segmented.getBytes(StandardCharsets.UTF_8)));
        }
    }
}
'''
