rm -rf resources;
rm -rf target;
rm -rf instance-generator/resources;
rm -rf instance-generator/target;
rm -rf tweaked-timenorm/resources;
rm -rf tweaked-timenorm/target;
mvn clean -U package;
mybroker/bin/artemis run;
java -cp instance-generator/target/instance-generator-5.0.0-SNAPSHOT-jar-with-dependencies.jar \
     org.apache.ctakes.core.pipeline.PiperFileRunner \
     -p org/apache/ctakes/timelines/pipeline/Process \
     -a mybroker/ \
     -v ~/miniconda3/envs/subtask1-multistep/ \
     -i ../input/notes/ \
     -d ../normalized_anafora/ \
     -o ../output/ \
     --pipPbj yes \
