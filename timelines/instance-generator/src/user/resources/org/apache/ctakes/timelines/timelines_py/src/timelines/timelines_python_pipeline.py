import warnings

from ctakes_pbj.component.pbj_receiver import PBJReceiver
from ctakes_pbj.pipeline.pbj_pipeline import PBJPipeline

from .timeline_annotator import TimelineAnnotator


warnings.filterwarnings("ignore")


def main():
    pipeline = PBJPipeline()
    receiver = PBJReceiver()
    annotator = TimelineAnnotator()
    pipeline.reader(receiver)
    pipeline.add(annotator)
    pipeline.initialize()
    pipeline.run()


main()
