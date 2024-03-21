java -cp instance-generator/target/instance-generator-5.0.0-SNAPSHOT-jar-with-dependencies.jar \
     org.apache.ctakes.core.pipeline.PiperFileRunner \
     -p org/apache/ctakes/timelines/pipeline/Preprocess \
     -i ../input/notes/ \
     -d ../input/anafora/ \
     -o ../normalized_anafora/ 
echo "NAME JEFF"
