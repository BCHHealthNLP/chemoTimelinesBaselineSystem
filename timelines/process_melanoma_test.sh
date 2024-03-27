java -cp instance-generator/target/instance-generator-5.0.0-SNAPSHOT-jar-with-dependencies.jar \
     org.apache.ctakes.core.pipeline.PiperFileRunner \
     -p org/apache/ctakes/timelines/pipeline/Process \
     -a mybroker/ \
     -v /usr/local/miniconda3/envs/subtask1 \
     -i ~/melanoma_annotated_test_filtered/ \
     -d ../normalized_anafora/melanoma_test/ \
     -o ../melanoma_all_test_filtered_unsummarized/ \
     --pipPbj yes \
