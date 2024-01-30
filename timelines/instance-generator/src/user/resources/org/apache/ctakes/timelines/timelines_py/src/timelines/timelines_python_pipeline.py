import warnings
import logging

from ctakes_pbj.component.pbj_receiver import start_receiver, PBJReceiver
from ctakes_pbj.component.pbj_sender import PBJSender
from ctakes_pbj.pipeline.pbj_pipeline import PBJPipeline

from .timeline_annotator import TimelineAnnotator


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def main():
    pipeline = PBJPipeline()
    receiver = PBJReceiver()
    annotator = TimelineAnnotator()
    pipeline.reader(receiver)
    pipeline.add(annotator)
    pipeline.initialize()
    pipeline.run()


main()
