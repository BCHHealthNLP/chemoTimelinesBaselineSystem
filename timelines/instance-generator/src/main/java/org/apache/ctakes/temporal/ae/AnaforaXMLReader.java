/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.commons.io.FilenameUtils;
import org.apache.ctakes.core.pipeline.PipeBitInfo;
import org.apache.ctakes.typesystem.type.structured.DocumentPath;
import org.apache.ctakes.typesystem.type.structured.SourceData;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.ctakes.core.util.ListFactory;
import org.apache.ctakes.typesystem.type.constants.CONST;
import org.apache.ctakes.typesystem.type.refsem.Event;
import org.apache.ctakes.typesystem.type.refsem.EventProperties;
import org.apache.ctakes.typesystem.type.relation.*;
import org.apache.ctakes.typesystem.type.textsem.*;
import org.apache.ctakes.typesystem.type.refsem.Time;
import org.apache.ctakes.typesystem.type.structured.SourceData;
import org.apache.ctakes.core.util.doc.SourceMetadataUtil;
import org.apache.log4j.Logger;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.jcas.cas.FSList;
import org.apache.uima.jcas.tcas.Annotation;
import org.apache.uima.resource.ResourceInitializationException;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;
import org.xml.sax.SAXException;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.*;
import java.util.stream.Stream;
import java.util.stream.Collectors;

@PipeBitInfo(
        name = "Anafora XML Reader (DeepPhe)",
        description = "Reads annotations from DeepPhe schema Anafora XML files in a directory.",
        role = PipeBitInfo.Role.SPECIAL,
        products = { PipeBitInfo.TypeProduct.IDENTIFIED_ANNOTATION }
)
public class AnaforaXMLReader extends JCasAnnotator_ImplBase {
  private static final Logger LOGGER = Logger.getLogger(AnaforaXMLReader.class);

  public static final String PARAM_ANAFORA_DIRECTORY = "anaforaDir";
  //one of 'pbj' or 'fw' (ctakes-pbj eval mode or format-writer)

  @ConfigurationParameter(
          name = PARAM_ANAFORA_DIRECTORY,
          description = "root directory of the Anafora-annotated files, with one subdirectory for "
                  + "each annotated file")
  private File anaforaDirectory;





  public static AnalysisEngineDescription getDescription( File anaforaDirectory )
          throws ResourceInitializationException {
    return AnalysisEngineFactory.createEngineDescription(
            AnaforaXMLReader.class,
            AnaforaXMLReader.PARAM_ANAFORA_DIRECTORY,
            anaforaDirectory
    );
  }

  @Override
  public void process(JCas jCas) throws AnalysisEngineProcessException {
      DocumentPath documentPath = JCasUtil.select( jCas, DocumentPath.class ).iterator().next();
      File jCasFilename = new File( documentPath.getDocumentPath() );
      String jCasBasename = FilenameUtils.getBaseName( jCasFilename.getAbsolutePath() );

      System.out.println( jCasBasename );
    File xmlFile = Arrays.stream(
                                 this
                                 .anaforaDirectory
                                 .listFiles() )
        .filter(
                n -> FilenameUtils
                .getBaseName( n.getAbsolutePath() )
                .startsWith( jCasBasename )
                )
        .findFirst()
        .orElse( null );

    if ( xmlFile == null ){
        System.err.println(xmlFile.getAbsolutePath());
        System.err.println(jCasFilename.getAbsolutePath());
    }

    try {
        processXmlFile( jCas, xmlFile );
    } catch (Exception e) {
        if( xmlFile != null ) System.err.println(xmlFile.getAbsolutePath() + " " + xmlFile.exists());
        System.err.println(jCasFilename.getAbsolutePath());
        throw new RuntimeException(e);
    }
  }


  private static void processXmlFile( JCas jCas, File xmlFile ) throws AnalysisEngineProcessException, FileNotFoundException, SAXException {
    // load the XML
    Element dataElem;
    try {
      dataElem = new SAXBuilder().build(xmlFile.toURI().toURL()).getRootElement();
    } catch (MalformedURLException e) {
      throw new AnalysisEngineProcessException(e);
    } catch (JDOMException e) {
      throw new AnalysisEngineProcessException(e);
    } catch (IOException e) {
      throw new AnalysisEngineProcessException(e);
    }

    int curEventId = 1;
    int curTimexId = 1;
    int curRelId = 1;
    int docLen = jCas.getDocumentText().length();
    final SourceData sourceData = SourceMetadataUtil.getOrCreateSourceData( jCas );
    for (Element annotationsElem : dataElem.getChildren("annotations")) {

      Map<String, Annotation> idToAnnotation = Maps.newHashMap();
      for (Element entityElem : annotationsElem.getChildren("entity")) {
        String id = removeSingleChildText(entityElem, "id", null);
        Element spanElem = removeSingleChild(entityElem, "span", id);
        String type = removeSingleChildText(entityElem, "type", id);
        Element propertiesElem = removeSingleChild(entityElem, "properties", id);

        // if ( type.equals("DOCTIME")) continue;
        // UIMA doesn't support disjoint spans, so take the span enclosing
        // everything
        int begin = Integer.MAX_VALUE;
        int end = Integer.MIN_VALUE;
        if ( !type.equals( "DOCTIME" ) ){
            try {
                //for (String spanString : spanElem.getText().split(";")) {
                for (String spanString : spanElem.getText().split(";")) {
                    String[] beginEndStrings = spanString.split(",");
                    // if (beginEndStrings.length != 2) {
                    //     error("span not of the format 'number,number'", id);
                    // }
                    int spanBegin = Integer.parseInt(beginEndStrings[0]);
                    int spanEnd = Integer.parseInt(beginEndStrings[1]);
                    if (spanBegin < begin && spanBegin >= 0) {
                        begin = spanBegin;
                    }
                    if (spanEnd > end && spanEnd <= docLen) {
                        end = spanEnd;
                    }
                }
                if(begin < 0 || end > docLen){
                    // error("Illegal begin or end boundary", id);
                    continue;
                }
            } catch (Exception e){
                throw new RuntimeException(e);
            }
        }


        Annotation annotation;
        if (type.equals("EVENT")) {
          String docTimeRel = removeSingleChildText(propertiesElem, "DocTimeRel", id);
          if (docTimeRel == null) {
            // error("no docTimeRel, assuming OVERLAP", id);
            docTimeRel = "OVERLAP";
          }
          String eventType = removeSingleChildText(propertiesElem, "Type", id);
          String degree = removeSingleChildText(propertiesElem, "Degree", id);
          String polarity = removeSingleChildText(propertiesElem, "Polarity", id);
          String contextualModality = removeSingleChildText(propertiesElem, "ContextualModality", id);
          String contextualAspect = removeSingleChildText(propertiesElem, "ContextualAspect", id);
          String permanence = removeSingleChildText(propertiesElem, "Permanence", id);
          EventMention eventMention = new EventMention(jCas, begin, end);
          Event event = new Event(jCas);
          EventProperties eventProperties = new EventProperties(jCas);
          eventProperties.setDocTimeRel(docTimeRel);
          eventProperties.setCategory(eventType);
          eventProperties.setDegree(degree);
          if (polarity.equals("POS")) {
            eventProperties.setPolarity(CONST.NE_POLARITY_NEGATION_ABSENT);
          } else if (polarity.equals("NEG")) {
            eventProperties.setPolarity(CONST.NE_POLARITY_NEGATION_PRESENT);
          } // else {
          //   error("polarity that was not POS or NEG", id);
          // }
          eventProperties.setContextualModality(contextualModality);
          eventProperties.setContextualAspect(contextualAspect);
          eventProperties.setPermanence(permanence);
          eventProperties.addToIndexes();
          event.setConfidence(1.0f);
          event.setDiscoveryTechnique(CONST.NE_DISCOVERY_TECH_GOLD_ANNOTATION);
          event.setProperties(eventProperties);
          event.setMentions(new FSArray(jCas, 1));
          event.setMentions(0, eventMention);
          event.addToIndexes();
          eventMention.setId(curEventId++);
          // eventMention.setAnaforaID(id);
          eventMention.setConfidence(1.0f);
          eventMention.setDiscoveryTechnique(CONST.NE_DISCOVERY_TECH_GOLD_ANNOTATION);
          eventMention.setEvent(event);
          eventMention.addToIndexes();
          annotation = eventMention;

        } else if (type.equals("TIMEX3")) {
          String timeClass = removeSingleChildText(propertiesElem, "Class", id);
          TimeMention timeMention = new TimeMention(jCas, begin, end);
          timeMention.setId(curTimexId++);
          timeMention.setTimeClass(timeClass);
          timeMention.addToIndexes();
          annotation = timeMention;
          if ( propertiesElem.getChildren( "normalizedExpression" ).size() > 0 ){
            String normalizedTimex = removeSingleChildText(propertiesElem, "normalizedExpression", id);
            if ( normalizedTimex != null ){
                Time time = timeMention.getTime();
                if (time == null){
                    time = new Time( jCas );
                    time.addToIndexes();
                }
                time.setNormalizedForm( normalizedTimex );
                timeMention.setTime( time );
            }
          }

        } else if (type.equals("DOCTIME")) {
            TimeMention timeMention = new TimeMention(jCas, begin, end);
            timeMention.setId(curTimexId++);
            timeMention.setTimeClass(type);
            timeMention.addToIndexes();
            annotation = timeMention;
            if ( propertiesElem.getChildren( "normalizedExpression" ).size() > 0 ){
                String normalizedTimex = removeSingleChildText(propertiesElem, "normalizedExpression", id);
                if ( normalizedTimex != null ){
                    sourceData.setSourceOriginalDate( normalizedTimex );
                }
            }
            annotation = null;
        }
        else if (type.equals("SECTIONTIME")) {
          TimeMention timeMention = new TimeMention(jCas, begin, end);
          timeMention.setId(curTimexId++);
          timeMention.setTimeClass(type);
          timeMention.addToIndexes();
          annotation = timeMention;

        } else if (type.equals("Markable")) {
          while(end >= begin && (jCas.getDocumentText().charAt(end-1) == '\n' || jCas.getDocumentText().charAt(end-1) == '\r')){
            end--;
          }
          Markable markable = new Markable(jCas, begin, end);
          markable.addToIndexes();
          annotation = markable;

        }
        else if (type.equals("DUPLICATE")) {
          // LOGGER.warn("Ignoring duplicate sections in annotations.");
          continue;
        } else {
          throw new UnsupportedOperationException("unsupported entity type: " + type);
        }

        // match the annotation to it's ID for later use
        if ( annotation != null ){
            idToAnnotation.put(id, annotation);
        }
        // make sure all XML has been consumed
        removeSingleChild(entityElem, "parentsType", id);
        if (!propertiesElem.getChildren().isEmpty() || !entityElem.getChildren().isEmpty()) {
          List<String> children = Lists.newArrayList();
          for (Element child : propertiesElem.getChildren()) {
            children.add(child.getName());
          }
          for (Element child : entityElem.getChildren()) {
            children.add(child.getName());
          }
          // error("unprocessed children " + children, id);
        }
      }
    }
  }


  private static Element removeSingleChild(Element elem, String elemName, String causeID) {
    Element child = getSingleChild(elem, elemName, causeID);
    elem.removeChildren(elemName);
    return child;
  }

  private static String removeSingleChildText(Element elem, String elemName, String causeID) {
    Element child = getSingleChild(elem, elemName, causeID);
    String text = child.getText();
    if (text.isEmpty()) {
      // error(String.format("an empty '%s' child", elemName), causeID);
      text = null;
    }
    elem.removeChildren(elemName);
    return text;
  }


  private static Element getSingleChild(Element elem, String elemName, String causeID) {
    List<Element> children = elem.getChildren(elemName);
    if (children.size() != 1) {
      // error(String.format("not exactly one '%s' child", elemName), causeID);
    }
    return children.size() > 0 ? children.get(0) : null;
  }

  private static List<Element> getAllChildren(Element elem,String elemName, String causeID){
    List<Element> children = new ArrayList<>();
    List<Element> temp = elem.getChildren(elemName);
    if (!(temp == null)){
      children = temp;
    }
    return children;
  }


  //Non-destructive editing
  private static String getSingleChildText(Element elem, String elemName, String causeID) {
    Element child = getSingleChild(elem, elemName, causeID);
    String text = null;
    if(child != null){
      text = child.getText();
    }
    if (text==null || text.isEmpty()) {
      text = null;
    }
    return text;
  }


  private static List<String> getAllChildrenText(Element elem, String elemName, String causeID) {
    List<Element> children = getAllChildren(elem, elemName, causeID);
    List<String> childrenTexts = new ArrayList<>();
    for (Element child : children) {
      String text = null;
      if (child != null) {
        text = child.getText();
      }
      if (!(text == null || text.isEmpty())) {
        childrenTexts.add(text);
      }
    }
    return childrenTexts;
  }



  private static Annotation getArgument(
          String id,
          Map<String, Annotation> idToAnnotation,
          String causeID) {
    Annotation annotation = idToAnnotation.get(id);
    // if (annotation == null) {
    //   error("no annotation with id " + id, causeID);
    // }
    return annotation;
  }

  private static Annotation getArgument(
          String id,
          Map<String, Annotation> idToAnnotation) {
    Annotation annotation = idToAnnotation.get(id);
    // if (annotation == null) {
    //   error("no annotation with id " + id, null);
    // }
    return annotation;
  }

  private static void error(String found, String id) {
    LOGGER.error(String.format("found %s in annotation with ID %s", found, id));
  }
}
