rm -rf resources;
rm -rf target;
rm -rf instance-generator/resources;
rm -rf instance-generator/target;
rm -rf tweaked-timenorm/resources;
rm -rf tweaked-timenorm/target;
mvn clean package;

java -cp instance-generator/target/instance-generator-5.0.0-SNAPSHOT-jar-with-dependencies.jar \
     org.apache.ctakes.core.pipeline.PiperFileRunner \
     -p org/apache/ctakes/timelines/pipeline/Timelines \
     -a  mybroker \
     -v /usr/local/miniconda3/envs/timelines-docker \
     -i ../input/ \
     -o ../output \
     -l org/apache/ctakes/dictionary/lookup/fast/bsv/Unified_Gold_Dev.xml \
     --pipPbj yes \
