java -cp instance-generator/target/instance-generator-5.0.0-SNAPSHOT-jar-with-dependencies.jar \
     org.apache.ctakes.core.pipeline.PiperFileRunner \
     -p org/apache/ctakes/timelines/pipeline/Timelines \
     -a mybroker/ \
     -v ~/miniconda3/envs/subtask1-multistep/ \
     -i ../input/notes/ \
     -d ../input/anafora/ \
     -o ../output/ \
     --pipPbj yes \
