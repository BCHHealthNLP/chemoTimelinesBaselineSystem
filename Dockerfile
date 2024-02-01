FROM redhat/ubi8:8.6

LABEL description="DeepPhe-Timelines Image"

# Set Maven version to be installed
ARG MAVEN_VERSION=3.8.6

WORKDIR /usr/src/app

# Copy everything else from host to image
COPY . .

# When trying to run "yum updates" or "yum install" the "system is not registered with an entitlement server" error message is given
# To fix this issue:
RUN echo $'[main]\n\
enabled=0\n\\n\
# When following option is set to 1, then all repositories defined outside redhat.repo will be disabled\n\
# every time subscription-manager plugin is triggered by dnf or yum\n\
disable_system_repos=0\n'\
>> /etc/yum/pluginconf.d/subscription-manager.conf

# Reduce the number of layers in image by minimizing the number of separate RUN commands
# Update packages
# Install the prerequisites
# Install which (otherwise 'mvn version' prints '/usr/share/maven/bin/mvn: line 93: which: command not found') and Java 8 via yum repository
# Download Maven tar file and install
# Install GCC, Git, Python 3.9, libraries needed for Python development
# Set default Python version for `python` command, `python3` already points to the newly installed Python3.9
# Upgrade pip, after upgrading, both pip and pip3 are the same version
# Download ActiveMQ Artemis 2.19.1 zip and extract. The final path: /usr/src/app/apache-artemis-2.19.1
# Create the Artemis broker 'mybroker'
# Clean all yum cache
RUN yum update -y && \
    yum install -y yum-utils && \
    yum install -y which java-11-openjdk java-11-openjdk-devel unzip && \
    curl -fsSL https://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz | tar xzf - -C /usr/share && \
    mv /usr/share/apache-maven-$MAVEN_VERSION /usr/share/maven && \
    ln -s /usr/share/maven/bin/mvn /usr/bin/mvn && \
    yum install -y gcc git python39 python39-devel && \
    alternatives --set python /usr/bin/python3.9 && \
    pip3 install --upgrade pip && \
    curl -LO https://archive.apache.org/dist/activemq/activemq-artemis/2.19.1/apache-artemis-2.19.1-bin.zip && \
    unzip apache-artemis-2.19.1-bin.zip && \
    apache-artemis-2.19.1/bin/artemis create mybroker --user deepphe --password deepphe --allow-anonymous && \
    yum clean all

# Set environment variables for Java and Maven
ENV JAVA_HOME /usr/lib/jvm/java
ENV M2_HOME /usr/share/maven
ENV maven.home $M2_HOME
ENV M2 $M2_HOME/bin
ENV PATH $M2:$PATH

# Change directory to build the rt_parser jar
WORKDIR /usr/src/app/timelines

#  hold-over until I debug this issue with nu-pip 
RUN pip install stomp.py dkpro-cassis transformers[torch] pandas tomli setuptools

RUN mvn -U clean package

# Execute the parser jar
CMD ["java", "-cp", \
"instance-generator/target/instance-generator-5.0.0-SNAPSHOT-jar-with-dependencies.jar", \
"org.apache.ctakes.core.pipeline.PiperFileRunner", \
"-p", "org/apache/ctakes/timelines/pipeline/Timelines", \
"-i", "/usr/src/app/input", \
"-o", "/usr/src/app/output", \
"-a", "/usr/src/app/mybroker", \
"-v", "/usr/bin/python", \
"-l", "org/apache/ctakes/dictionary/lookup/fast/bsv/Unified_Gold_Dev.xml", \
"--pipPbj", "yes"]