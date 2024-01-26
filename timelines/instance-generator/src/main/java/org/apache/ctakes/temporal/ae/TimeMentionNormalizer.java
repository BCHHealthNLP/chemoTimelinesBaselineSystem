package org.apache.ctakes.temporal.ae;

import org.apache.commons.io.FilenameUtils;
import org.apache.ctakes.core.cc.AbstractJCasFileWriter;
import org.apache.ctakes.core.pipeline.PipeBitInfo;
import org.apache.ctakes.core.util.annotation.IdentifiedAnnotationUtil;
import org.apache.ctakes.core.util.annotation.OntologyConceptUtil;
import org.apache.ctakes.core.util.doc.SourceMetadataUtil;
import org.apache.ctakes.typesystem.type.constants.CONST;
import org.apache.ctakes.typesystem.type.refsem.Event;
import org.apache.ctakes.typesystem.type.refsem.EventProperties;
import org.apache.ctakes.typesystem.type.refsem.UmlsConcept;
import org.apache.ctakes.typesystem.type.structured.SourceData;
import org.apache.ctakes.typesystem.type.structured.DocumentPath;
import org.apache.ctakes.typesystem.type.textsem.EventMention;
import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;
import org.apache.ctakes.typesystem.type.textsem.MedicationMention;
import org.apache.ctakes.typesystem.type.textsem.TimeMention;
// import org.apache.ctakes.typesystem.type.refsem.Date;
import org.apache.ctakes.typesystem.type.refsem.Time;
import org.apache.ctakes.typesystem.type.structured.SourceData;
import org.apache.log4j.Logger;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.clulab.timenorm.scfg.Temporal;
import org.clulab.timenorm.scfg.TemporalExpressionParser;
import org.clulab.timenorm.scfg.TimeSpan;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import me.tongfei.progressbar.*;
import com.google.common.util.concurrent.TimeLimiter;
import com.google.common.util.concurrent.SimpleTimeLimiter;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
@PipeBitInfo(
        name = "TimeMentionNormalizer",
        description = "Normalizes time expressions",
        dependencies = { PipeBitInfo.TypeProduct.EVENT, PipeBitInfo.TypeProduct.TIMEX }
)


public class TimeMentionNormalizer extends org.apache.uima.fit.component.JCasAnnotator_ImplBase {
    static private final Logger LOGGER = Logger.getLogger( "TimeMentionNormalizer" );

    // generalize to multiple at some point, check how the dictionary system
    // does it with filtering based on syntactic category
    public static final String PARAM_TUIS = "tuis";

    @ConfigurationParameter(
                            name = PARAM_TUIS,
                            description = "The way we store files for processing.  Aligned pair of directories ",
                            defaultValue = "T061",
                            mandatory = false
    )
    private String tuis;

    public static final String PARAM_TIMEOUT = "timeout";
    public static final int DEFAULT_TIMEOUT = 5;
    @ConfigurationParameter(
                            name = PARAM_TIMEOUT,
                            description = "The way we store files for processing.  Aligned pair of directories ",
                            mandatory = false
    )
    private int timeout = DEFAULT_TIMEOUT;
    private Set<String> tuiSet;

    static private final TemporalExpressionParser normalizer = TemporalExpressionParser.en();
    static private final TimeLimiter timeLimiter = SimpleTimeLimiter.create(Executors.newSingleThreadExecutor());

    @Override
    public void initialize( UimaContext context ) throws ResourceInitializationException {
        super.initialize( context );
        this.tuiSet = new HashSet<String>();
        final String[] tuiArr = tuis.split( "," );
        for ( String tui : tuiArr ) {
            this.tuiSet.add( tui.toUpperCase() );
        }

        final Object _timeout = context.getConfigParameterValue( PARAM_TIMEOUT );
        if ( _timeout != null ) {
            this.timeout = parseInt( _timeout, PARAM_TIMEOUT, this.timeout );
        }
        LOGGER.info( "Using timeout: " + this.timeout );
    }

    @Override
    public void process( JCas jCas ) throws AnalysisEngineProcessException {
        final SourceData sourceData = SourceMetadataUtil.getOrCreateSourceData( jCas );
        final String docTime = sourceData.getSourceOriginalDate();
        DocumentPath documentPath = JCasUtil.select( jCas, DocumentPath.class ).iterator().next();
        final String fileName = FilenameUtils.getBaseName( documentPath.getDocumentPath() );
        if (this.tuis != null && !this.tuis.trim().isEmpty()){
            boolean hasRelevantTUIs = JCasUtil
                .select( jCas, EventMention.class )
                .stream()
                .map( OntologyConceptUtil::getUmlsConcepts )
                .flatMap( Collection::stream )
                .map( UmlsConcept::getTui )
                .anyMatch( tui -> this.tuiSet.contains( tui ) );

            if ( !hasRelevantTUIs ){
                LOGGER.info(fileName + " : no events with the provided TUIs " + this.tuis + "skipping to save time");
                return;
            }
        }

        TimeSpan _DCT = null;
        if ( docTime == null || docTime.isEmpty() ){
            LOGGER.warn( fileName + ": Empty Document Creation Time" );
        } else {
            String[] docTimeComponents = docTime.split("-");
            // properly generated
            if (docTimeComponents.length == 3) {
                _DCT = TimeSpan.of(
                                  Integer.parseInt(docTimeComponents[0]),
                                  Integer.parseInt(docTimeComponents[1]),
                                  Integer.parseInt(docTimeComponents[2]));
            } else {
                // DocTimeApproximator generated
                _DCT = TimeSpan.of(
                                  Integer.parseInt(docTime.substring(0, 4)),
                                  Integer.parseInt(docTime.substring(4, 6)),
                                  Integer.parseInt(docTime.substring(6, 8)));
            }
        }
        final TimeSpan DCT = _DCT;
        List<TimeMention> timeMentions = JCasUtil
            .select( jCas, TimeMention.class )
            .stream()
            .collect( Collectors.toList() );

        for ( TimeMention timeMention : ProgressBar.wrap( timeMentions, fileName + ": Normalizing TimeMentions" ) ){
            normalize( jCas, DCT, fileName, timeMention );
        }
    }

    private void normalize( JCas jCas, TimeSpan DCT, String fileName, TimeMention timeMention ){
        String normalizedTimex = getTimeML( DCT, timeMention, fileName );
        if ( normalizedTimex.length() > 0 ){
            Time time = timeMention.getTime();
            if (time == null){
                time = new Time( jCas );
                time.addToIndexes();
            }
            time.setNormalizedForm( normalizedTimex );
            timeMention.setTime( time );
        }
    }

    private String getTimeML( TimeSpan DCT, TimeMention timeMention, String fileName ){
        String rawTimeMention = timeMention.getCoveredText();
        String[] rawDateElements = rawTimeMention.split("/");
         List<Integer> dateElements = new ArrayList<>();
        for ( String dateElement : rawDateElements ){
            try {
                int elem = Integer.parseInt( dateElement );
                dateElements.add( elem );
            } catch ( Exception ignored ){}
        }
        // can also do this in a way that grabs stragglers
        if ( dateElements.size() == 3 && rawDateElements.length == 3 ){
            // avoiding TimeNorm's issues with component order
            // ambiguity since these notes were all generated
            // at American hospitals and therefore modulo mistakes
            // will all use the American convention
            int month = dateElements.get( 0 );
            int date = dateElements.get( 1 );
            int raw_year = dateElements.get( 2 );
            int year;
            if ( rawDateElements[2].length() == 2 ){
                year = raw_year + 2000;
            } else {
                year = raw_year;
            }
            TimeSpan parsedDate = null;
            try{
                parsedDate = TimeSpan.of( year, month, date );
                String dateMLValue = parsedDate.timeMLValue();
                // LOGGER.info( fileName + ": successfully rule-parsed " + rawTimeMention + " as " + dateMLValue );
                return dateMLValue;
            } catch ( Exception ignored ){
                // LOGGER.error( fileName + ": failed to do a rule-based parse for " + rawTimeMention );
            }
        }
        String unnormalizedTimex = String.join(" ", timeMention.getCoveredText().split("\\s"));
        Temporal normalizedTimex = null;
        try{
            try{
                normalizedTimex = timeLimiter
                    .callUninterruptiblyWithTimeout(
                        () -> normalizer.parse( unnormalizedTimex, DCT ).get(),
                        timeout,
                        TimeUnit.SECONDS );
            } catch ( Exception ignored ){
                LOGGER.error( fileName + ": Timenorm could not parse timex " + timeMention.getCoveredText() + " in " + timeout + " seconds or less");
                return "";
            }
        } catch ( Exception ignored ){
            LOGGER.error( fileName + ": Timenorm failed to normalize timex " + unnormalizedTimex );
            return "";
        }
        return normalizedTimex.timeMLValue();
    }

    // Code due to Sean
    static private int parseInt( final Object value, final String name, final int defaultValue ) {
        if ( value instanceof Integer ) {
            return (Integer)value;
        } else if ( value instanceof String ) {
            try {
                return Integer.parseInt( (String)value );
            } catch ( NumberFormatException nfE ) {
                LOGGER.warn( "Could not parse " + name + " " + value + " as an integer" );
            }
        } else {
            LOGGER.warn( "Could not parse " + name + " " + value + " as an integer" );
        }
        return defaultValue;
    }

}
