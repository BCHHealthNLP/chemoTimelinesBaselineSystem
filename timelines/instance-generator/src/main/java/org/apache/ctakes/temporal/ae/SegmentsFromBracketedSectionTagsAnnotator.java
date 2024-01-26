package org.apache.ctakes.temporal.ae;

import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.jcas.JCas;
import org.apache.ctakes.typesystem.type.textspan.Segment;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

// from Sean or Tim originally, I think it's still actually in ctakes temporal somewhere but idk
public class SegmentsFromBracketedSectionTagsAnnotator extends JCasAnnotator_ImplBase {
    private static Pattern SECTION_PATTERN = Pattern.compile(
            "(\\[start section id=\"?(.*?)\"?\\]).*?(\\[end section id=\"?(.*?)\"?\\])",
            Pattern.DOTALL );

    @Override
    public void process( JCas jCas ) throws AnalysisEngineProcessException {
        boolean foundSections = false;
        Matcher matcher = SECTION_PATTERN.matcher( jCas.getDocumentText() );
        while ( matcher.find() ) {
            Segment segment = new Segment( jCas );
            segment.setBegin( matcher.start() + matcher.group( 1 ).length() );
            segment.setEnd( matcher.end() - matcher.group( 3 ).length() );
            segment.setId( matcher.group( 2 ) );
            segment.addToIndexes();
            foundSections = true;
        }
        if ( !foundSections ) {
            Segment segment = new Segment( jCas );
            segment.setBegin( 0 );
            segment.setEnd( jCas.getDocumentText().length() );
            segment.setId( "SIMPLE_SEGMENT" );
            segment.addToIndexes();
        }
    }
}
