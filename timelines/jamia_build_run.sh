rm -rf resources;
rm -rf target;
rm -rf instance-generator/resources;
rm -rf instance-generator/target;
rm -rf tweaked-timenorm/resources;
rm -rf tweaked-timenorm/target;
mvn -U clean package;

java -cp instance-generator/target/instance-generator-5.0.0-SNAPSHOT-jar-with-dependencies.jar \
     org.apache.ctakes.core.pipeline.PiperFileRunner \
     -p org/apache/ctakes/timelines/pipeline/Timelines \
     -a  mybroker \
     -v ~/miniconda3/envs/timelines-docker \
     -t ~/HDD/jamia-system-models/tlink_eb/ \
     -m ~/HDD/jamia-system-models/tagger_pmb/ \
     -i ../input/ \
     -o ../output \
     --pipPbj yes \
