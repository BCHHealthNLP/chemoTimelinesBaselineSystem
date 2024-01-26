import warnings
import logging

from ctakes_pbj.component.pbj_receiver import start_receiver, PBJReceiver
from ctakes_pbj.component.pbj_sender import PBJSender
from ctakes_pbj.pipeline.pbj_pipeline import PBJPipeline

from .timeline_delegator import TimelineDelegator


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def main():
    print("In main of PBJ annotator driver")
    # Create a new PBJ Pipeline, add a class that interacts with cNLPT to add Negation to Events.
    pipeline = PBJPipeline()
    receiver = PBJReceiver()
    annotator = TimelineDelegator()
    # sender = PBJSender()
    pipeline.reader(receiver)
    pipeline.add(annotator)
    # Add a PBJ Sender to the end of the pipeline to send the processed cas back to cTAKES and initialize the pipeline.
    # pipeline.add(sender)
    pipeline.initialize()
    pipeline.run()
    # Start a PBJ receiver to accept cas objects from Artemis and process them in the pipeline.
    # start_receiver(pipeline)


main()
