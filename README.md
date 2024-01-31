# ChemoTimelines Docker

Dockerizable source code for the baseline system for the [Chemotherapy Treatment Timelines Extraction from the Clinical Narrative](https://sites.google.com/view/chemotimelines2024/task-descriptions) shared task.

## Warning

This is research code which depends on other research code.  None of which is shrink wrapped.  Run at your own risk and do not use in any kind of clinical decision making context.

While operational there are known issues in the code's dependencies which are still being resolved.

## Core dependencies

There are three main separate software packages that this code uses:
- [Apache cTAKES](https://github.com/apache/ctakes)
- [CLU Lab Timenorm](https://github.com/clulab/timenorm)
- [Huggingface Transformers](https://huggingface.co/docs/transformers/index)


cTAKES contains several tools for text engineering and information extraction with a focus on clinical text, it makes heavy use of [Apache UIMA](https://uima.apache.org).
Within cTAKES the main module which drives this code is the cTAKES [Python Bridge to Java](https://github.com/apache/ctakes/tree/main/ctakes-pbj).
While cTAKES is written in Java, the Python Bridge to Java (*ctakes-pbj*) allows use of Python code to process text artifacts the same way one can do with Java code in cTAKES.  *ctakes-pbj* accomplishes this by passing text artifacts and their extracted information between the relevant Java and Python processes using [DKPro cassis]( https://github.com/dkpro/dkpro-cassis) for serialization, [Apache ActiveMQ]( https://activemq.apache.org) for message brokering, and [stomp.py](https://github.com/jasonrbriggs/stomp.py) for Python-side receipt from and transmission to ActiveMQ.

Timenorm provides methods for identifying and normalizing date and time expressions.  We use a customized version (included as a maven module) where we change a heuristic for approximate dates to better address the needs of the timelines project.

We used Huggingface Transformers for training the TLINK model, and use their [Pipelines interface](https://huggingface.co/docs/transformers/main_classes/pipelines) for loading the model for inference. We use the [Huggingface Hub](https://huggingface.co/HealthNLP) for model storage.

## Recommended Hardware

A CUDA capable GPU with at least 500mb of VRAM is preferred for running the TLINK classifier, but with sufficient standard RAM the model can be run on CPU.  Outside of the Docker nothing needs to be done to effect this change, if however you want to run the Docker on a machine with no GPU ( or to disable GPU use ) then comment out the following lines in `docker-compose.yml`:
```
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
This also means you would not need the NVIDIA container toolkit.

## Classifiers

Our classifiers (currently only using TLINK) are accessible at https://huggingface.co/HealthNLP.  By default the code downloads and loads the TLINK classifier from the Huggingface repository.

## High-level system description

Each document is annotated with paragraphs, sentences, and tokens by cTAKES.  The cTAKES dictionary module searches over the tokens for spans which match chemotherapy mentions in the gold data annotations (in this regard we are using gold entities for chemos, although *not* for time expressions).  Then a cTAKES SVM-based tagger finds token spans which correspond to temporal expressions, and we use Timenorm to normalize them to ISO format.  Finally, we create instances of chemotherapy and normalized date pairs, and pass them to a PubMedBert-based classifier which identifies the temporal relationship between the paired mentions.  Finally the code outputs a file with all the classified instances organized by patient and filename, with unique identifiers for each chemotherapy mention and temporal expression.


## Overview of Docker dependencies

- [Docker Engine](https://docs.docker.com/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Data directories

There are three mounted directories:

- *Input*: The collection of notes in a shared task cancer type cohort
- *Processing*: Timeline information extraction over each note within cTAKES, aggregation of results by patient identifier
- *Output*: Aggregated unsummarized timelines information in a `tsv` file

## Build a Docker image

Under the project root directory ( you may need to use `sudo` ) run:

```
docker compose build --no-cache
```

## Start a container

You may need `sudo` here as well:

```
docker compose up
```
## Critical operation instruction

Due to a current issue in the inter process communication, the process will finish writing but not close itself.  So when you see `Writing results for ...` followed by `Finished writing...` close the process via `ctrl+c`.  This is the case both for running the system inside or outside of a Docker image.

## Running the system outside of a Docker image

This is for the most part actually how we have ran the system during development, and can be resorted to in the event of issues with creating or running a Docker image.  Use the following steps for setup:
- Make sure you have Java JDK 8  (we use OpenJDK) and the latest version of maven installed and that Java 8 is set as your default system Java
- Create a conda 3.9 environment with `conda create -n timelines python=3.9`
- Change directory into `timelines` under the project root
- Create an ActiveMQ broker named `mybroker` in your current directory via:
```
curl -LO https://archive.apache.org/dist/activemq/activemq-artemis/2.19.1/apache-artemis-2.19.1-bin.zip && \
unzip apache-artemis-2.19.1-bin.zip && \
apache-artemis-2.19.1/bin/artemis create mybroker --user deepphe --password deepphe --allow-anonymous
```
- (temporary fix until we fix the PBJ and timelines dependencies issue) Install the system's Python dependencies via:
```
pip install stomp.py dkpro-cassis transformers[torch] pandas tomli setuptools
```
- Finally create the Java executable Jars with maven:
```
mvn -U clean package
```
If you run into issues with Torch, you might want to look into finding the Torch setup [most appropriate for your configuration](https://pytorch.org/get-started/locally/) and install it via `conda`.

Finally, assuming everything compiled and your *input* folder is populated you can run the system via:
```
mybroker/bin/artemis run &
```
Then, once the ActiveMQ broker is finished starting:
```
java -cp instance-generator/target/instance-generator-5.0.0-SNAPSHOT-jar-with-dependencies.jar \
     org.apache.ctakes.core.pipeline.PiperFileRunner \
     -p org/apache/ctakes/timelines/pipeline/Timelines \
     -a  mybroker \
     -v <path to your conda env, on my machine it's in /usr/local/miniconda3/envs> \
     -i ../input/ \
     -o ../output \
     -l org/apache/ctakes/dictionary/lookup/fast/bsv/Unified_Gold_Dev.xml \
     --pipPbj yes \
```
And assuming successful processing your `tsv` file should be in the *output* folder of the project root directory.  Finally, stop the ActiveMQ via:
```
mybroker/bin/artemis stop
```
## Input and output structure

Given the structure of the summarized gold timelines and the shared task data, the Docker assumes that the input in the `input`
folder will take the form of a collection of notes comprising all the patients of a given cancer type cohort.  Assuming successful processing, the output file will be a tab separated value (`tsv`) file in the `output` folder.
The file will have the columns:
```
DCT	patient_id	chemo_text	chemo_annotation_id	normed_timex	timex_annotation_id	tlink	note_name	tlink_inst
```
And each row corresponds to a TLINK classification instance from a given file.  In each row:
 - The `DCT` cell will hold the document creation date of the file which is the source of the instance
 - The `patient_id` cell will hold the patient identifier of the file which is the source of the instance
 - `chemo_text` cell will hold the raw text of the chemotherapy mention in the instance as it appears in the note
 - `chemo_annotation_id` assigns the chemotherapy mention in the previous cell a unique identifier (at the token rather than the type level)
 - `normed_timex` will hold the normalized version of the time expression in the tlink instance
 - `timex_annotation_id` assigns the time expression in the previous cell a unique identifier (at the token rather than the type level)
 - `note_name` holds the name of the corresponding file
 - `tlink_inst` holds the full chemotherapy timex pairing instance that was fed to the classifier (mostly for debugging purposes)


## Architecture

We use two maven modules, one for the Java and Python annotators relevant to processing the clinical notes, and the other which has the customized version of Timenorm.  There are not many files and their classpaths are not especially important for understanding, but more so for customization.

For further documentation, please see the [cTAKES wiki](https://github.com/apache/ctakes/wiki), in particular for [cTAKES PBJ](https://github.com/apache/ctakes/wiki/pbj_intro).  If you run into issues in your attempts to customize the code, please use the contact information at the bottom of the README, or if appropriate, submit an issue.

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

## The piper file

The piper file at `org/apache/ctakes/timelines/pipeline/Timelines` describes the flow logic of the information extraction, e.g. the annotators involved, the order in which they are run, as well as their configuration parameters.

The contents of the piper file by default are:
```
package org.apache.ctakes.timelines

set SetJavaHome=no
set ForceExit=no

load PbjStarter

add PythonRunner Command="-m pip install resources/org/apache/ctakes/timelines/timelines_py" Wait=yes

set TimelinesSecondStep=timelines.timelines_python_pipeline

add PythonRunner Command="-m $TimelinesSecondStep -rq JavaToPy -o $OutputDirectory"

set minimumSpan=2
set exclusionTags=“”

// Just the components we need from DefaultFastPipeline

// Write nice big banners when ctakes starts and finishes.
set WriteBanner=yes

// Load a simple token processing pipeline from another pipeline file
load DefaultTokenizerPipeline

// Add non-core annotators
add ContextDependentTokenizerAnnotator
// Dictionary module requires tokens so needs to be loaded after the tokenization stack
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
set TimelinesSecondStep=timelines.timelines_python_pipeline

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

## Core Python processing annotator

The core Python logic is in the file:
```
timelines/instance-generator/src/user/resources/org/apache/ctakes/timelines/timelines_py/src/timelines/timelines_annotator.py
```
Like the Java annotators the Python annotator implements a `process` method which is the core driver of the annotator for processing each note's contents.  The raw output for the whole cancer type cohort is collected and written to TSV on disk in the `collection_process_complete` method.

## Questions and technical issues

Please contact [Eli Goldner](mailto:eli.goldner@childrens.harvard.edu?subject=Timelines%20Docker%20Issue/Question) for non code-level issues or questions.  For issues in the code please open an issue through the repository page on GitHub.
