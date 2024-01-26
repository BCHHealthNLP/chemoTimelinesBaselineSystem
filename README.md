# ChemoTimelines Docker

Dockerizable source code for the author system for the [Chemotherapy Treatment Timelines Extraction from the Clinical Narrative](https://sites.google.com/view/chemotimelines2024/task-descriptions) shared task.

## Overview of Docker dependencies

- [Docker Engine](https://docs.docker.com/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

The image (by default) requires three folders for the TLINK, DTR, and contexual modality Huggingface classifiers at the top of the repository with the respective names `tlink`, `dtr`, and `conmod`.

## Data directories

There are three mounted directories:

- *Input*: Each subfolder should be a patient identifier, the contents of the subfolder should be the patient's notes
- *Processing*: Timeline information extraction over each note within cTAKES, aggregation of results by patient identifier
- *Output*: Aggregated timelines information, one `tsv` file per patient identifier 

## Build docker image

Under the project root directory ( you may need to use `sudo` ):

```
docker compose build --no-cache
```


## Start a container

You may need `sudo` here as well:

```
docker compose up
```

## Input and output structure

Given the structure of the summarized gold timelines and the shared task data, the Docker assumes that the input in the `input`
folder will take the form of a collection of notes comprising all the patients of a given cancer type cohort (for the shared task one of melanoma or ovarian or breast cancers), the base filenames of which will correspond to the scheme:
```
<patient identifier>_<four digit year>_<two digit month>_<two digit date>
```
Where the year month and date correspond to the creation time of the file.  
All the files in the shared task dataset follow this schema so for our data there is nothing you need to do. 

Assuming successful processing, the output file will be a tab separated value (`tsv`) file in the `output` folder.
The file will have the columns:
```
DCT	patient_id	chemo_text	chemo_annotation_id	normed_timex	timex_annotation_id	tlink	note_name	tlink_inst
```
And each row corresponds to a TLINK classification instance from a given file.  In each row:
 - The `DCT` cell will hold the document creation time/date of the file which is the source of the instance
 - The `patient_id` cell will hold the patient identifier of the file which is the source of the instance
 - `chemo_text` cell will hold the raw text of the chemotherapy mention in the instance as it appears in the note
 - `chemo_annotation_id` assigns the chemotherapy mention in the previous cell a unique identifier (at the token rather than the type level)
 - `normed_timex` will hold the normalized version of the time expression in the tlink instance
 - `timex_annotation_id` assigns the time expression in the previous cell a unique identifier (at the token rather than the type level)
 - `note_name` holds the name of the corresponding file (technically redundant if your files correspond to specification)
 - `tlink_inst` holds the full chemotherapy timex pairing instance that was fed to the classifier (mostly for debugging purposes)

## Core dependencies

There are three main separate software packages that this code uses:
- [Apache cTAKES](https://github.com/apache/ctakes)
- [CLU Lab Timenorm](https://github.com/clulab/timenorm)
- [Huggingface Transformers](https://huggingface.co/docs/transformers/index)


cTAKES contains several tools for text engineering and information extraction with a focus on clinical text, it is based on [Apache UIMA](https://uima.apache.org).
Within cTAKES the main module which drives this code is the cTAKES [Python Bridge to Java](https://github.com/apache/ctakes/tree/main/ctakes-pbj).
While cTAKES is written in Java, the Python Bridge to Java (*ctakes-pbj*) allows for use of python code to process text artifacts the same way one can do 
with Java within cTAKES.  *ctakes-pbj* accomplishes this by passing text artifacts and their annotated information between the relevant Java and Python processes 
using [DKPro cassis]( https://github.com/dkpro/dkpro-cassis) for serialization, [Apache ActiveMQ]( https://activemq.apache.org) for message brokering, and [stomp.py](https://github.com/jasonrbriggs/stomp.py) for Python-side receipt from and transmission to ActiveMQ.  

Timenorm provides methods for identifying normalizing date and time expressions.  We use a customized version (included as a maven module) where we change a heuristic for approximate dates.

We used Huggingface Transformers for training the TLINK model, and use their [Pipelines interface](https://huggingface.co/docs/transformers/main_classes/pipelines) for loading the model for inference.


## Architecture

We use two maven modules, one for the Java and Python annotators relevant to processing the clinical notes, and the other which has the customized version of Timenorm.

### Core command

The central command in the Docker (you can also run it outside the Docker with the appropriate dependencies):
```
java -cp instance-generator/target/instance-generator-5.0.0-SNAPSHOT-jar-with-dependencies.jar \
     org.apache.ctakes.core.pipeline.PiperFileRunner \
     -p org/apache/ctakes/timelines/pipeline/Timelines \
     -a  mybroker \
     -i ../input/ \
     -o ../output \
     -l org/apache/ctakes/dictionary/lookup/fast/bsv/Unified_Gold_Dev.xml \
     --pipPbj yes \
```
The `org.apache.ctakes.core.pipeline.PiperFileRunner` class is the entry point. `-a mybroker` points to the ActiveMQ broker for the process (you can see how to set one up in the Dockerfile). 

### (Optional) running the core command outside of the Docker

Add to the core command `-v <path to conda environment>** to run with a specified conda environment.  Ideally, cTAKES should start the ActiveMQ broker by itself when you run the command.  However we have had to start the broker ourselves before running the command via 
```
mybroker/bin/artemis run &
```
And stopping it after the run with:
```
mybroker/bin/artemis stop
```
## The piper file

The piper file at `org/apache/ctakes/timelines/pipeline/Timelines` describes the flow logic of the information extraction, e.g. the annotators involved, the order in which they are run, as well as their configuration parameters.

The contents of the piper file by default are:
```
package org.apache.ctakes.timelines

set SetJavaHome=no
set ForceExit=no

load PbjStarter

add PythonRunner Command="-m pip install resources/org/apache/ctakes/timelines/timelines_py" Wait=yes

set TimelinesSecondStep=timelines.timelines_pipeline

add PythonRunner Command="-m $TimelinesSecondStep -rq JavaToPy -o $OutputDirectory"

set minimumSpan=2
set exclusionTags=“”

// Just the components we need from DefaultFastPipeline
set WriteBanner=yes

// Load a simple token processing pipeline from another pipeline file
load DefaultTokenizerPipeline

// Add non-core annotators
add ContextDependentTokenizerAnnotator
load DictionarySubPipe

add BackwardsTimeAnnotator classifierJarPath=/org/apache/ctakes/temporal/models/timeannotator/model.jar
add DCTAnnotator
add TimeMentionNormalizer timeout=10

add PbjJmsSender SendQueue=JavaToPy SendStop=yes
```

To break down what's happening here in broad strokes:
```
package org.apache.ctakes.timelines

set SetJavaHome=no
set ForceExit=no

load PbjStarter

add PythonRunner Command="-m pip install resources/org/apache/ctakes/timelines/timelines_py" Wait=yes
```
This sets up the necessary environment variables and installs the relevant Python code as well as its dependencies to the Python environment.
```
set TimelinesSecondStep=timelines.timelines_pipeline

add PythonRunner Command="-m $TimelinesSecondStep -rq JavaToPy -o $OutputDirectory"
```
This starts the Python annotator and has it wait on the ArtemisMQ receive queue for incoming CASes.    
```
set minimumSpan=2
set exclusionTags=“”

// Just the components we need from DefaultFastPipeline
set WriteBanner=yes

// Load a simple token processing pipeline from another pipeline file
load DefaultTokenizerPipeline

// Add non-core annotators
add ContextDependentTokenizerAnnotator
load DictionarySubPipe
```
`minimumSpan` and `exclusionTags` are both configuration parameters for the dictionary lookup module, we don't exclude any parts of speech for lookup and want only to retrieve turns of at least two characters.  The `DefaultTokenizerPipeline` annotates each CAS for paragraphs, sentences, and tokens.  The `ContextDependentTokenizerAnnotator` depends on annotated base tokens and identifies basic numerical expressions for dates and times.  The `DictionarySubPipe` module loads the dictionary configuration XML provided with the `-l` tag in the execution of the main Jar file.          
```
add BackwardsTimeAnnotator classifierJarPath=/org/apache/ctakes/temporal/models/timeannotator/model.jar
add DCTAnnotator
add TimeMentionNormalizer timeout=10
```
`BackwardsTimeAnnotator` invokes a SVM-based classifier to identify additional temporal expressions.  `DCTAnnotator` identifies the document creation time for each note, which is needed for normalizing relative temporal expressions.  `TimeMentionNormalizer` invokes the Timenorm context free grammar parser to normalize all temporal expressions possible.  Some default behaviors with this are worth noting, firstly, to save processing time, by default we skip normalizing temporal expressions from notes which do not have any chemotherapy mentions, secondly, due to some issues with processing time for noisy temporal expressions, there is a timeout parameter for when to quit an attempted normalization parse.  Unless specified the timeout defaults to five seconds.

And finally:
```
add PbjJmsSender SendQueue=JavaToPy SendStop=yes
```
Sends the CASes which have been processed by the Java annotators to the Python annotator via the ActiveMQ send queue.

## Questions and technical issues

Please contact [Eli Goldner](mailto:eli.goldner@childrens.harvard.edu?subject=Timelines%20Docker%20Issue/Question) for non code-level issues or questions.  For issues in the code please open an issue through the repository page on GitHub.
