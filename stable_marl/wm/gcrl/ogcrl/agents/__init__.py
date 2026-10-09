from .crl import CRLAgent
from .gcbc import GCBCAgent
from .gciql import GCIQLAgent
from .gcivl import GCIVLAgent
from .hiql import HIQLAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
)
