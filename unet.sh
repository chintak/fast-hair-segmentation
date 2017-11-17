#!/bin/bash

set -e

if [ $# -eq 0 ]; then
    echo "sh unet.sh u_config_num"
    exit;
fi

set -x
U=u"$1"
set +x
echo "======================"
echo "Model #1"
echo "======================"
set -x

L=l0
F="$U"featsl3"$L"
M=optim
DT=LFW/u"$L"hair_"$F".txt.train
DV=LFW/u"$L"hair_"$F".txt.validation
MNAME0a=models/"$M"_"$F"_001_"$L".model
MNAME0b=models/"$M"_"$F"_002_"$L".model
MNAME0=$MNAME0a

if [ ! -f "$DT" ]; then
    python data.py Train $F -o "$DT"
    python data.py Validation $F -o "$DV"
fi
if [ ! -f "$MNAME0a" ]; then
    python trainer.py -t "$DT" -v "$DV" "$MNAME0a"
fi
if [ ! -f "$MNAME0b" ]; then :
    # python trainer.py -t "$DT" -v "$DV" -c "$MNAME0a" "$MNAME0b"
fi

set +x
echo "======================"
echo "Model #2"
echo "======================"
set -x

L=l1
F="$U"featsl3"$L"
M=optim
DT=LFW/u"$L"hair_"$F".txt.train
DV=LFW/u"$L"hair_"$F".txt.validation
MNAME1a=models/"$M"_"$F"_001_"$L".model
MNAME1b=models/"$M"_"$F"_002_"$L".model
MNAME1=$MNAME1a

if [ ! -f "$DT" ]; then
    python data.py Train $F -o "$DT" -u "$MNAME0"
    python data.py Validation $F -o "$DV" -u "$MNAME0"
fi
if [ ! -f "$MNAME1a" ]; then
    python trainer.py -t "$DT" -v "$DV" "$MNAME1a"
fi
if [ ! -f "$MNAME1b" ]; then :
    # python trainer.py -t "$DT" -v "$DV" -c "$MNAME1a" "$MNAME1b"
fi

set +x
echo "======================"
echo "Model #3"
echo "======================"
set -x

L=l2
F="$U"featsl3"$L"
M=optim
DT=LFW/u"$L"hair_"$F".txt.train
DV=LFW/u"$L"hair_"$F".txt.validation
MNAME2a=models/"$M"_"$F"_001_"$L".model
MNAME2b=models/"$M"_"$F"_002_"$L".model
MNAME2=$MNAME2b

if [ ! -f "$DT" ]; then
    python data.py Train $F -o "$DT" -u "$MNAME1"
    python data.py Validation $F -o "$DV" -u "$MNAME1"
fi
if [ ! -f "$MNAME2a" ]; then
    python trainer.py -t "$DT" -v "$DV" "$MNAME2a"
fi
if [ ! -f "$MNAME2b" ]; then
    python trainer.py -t "$DT" -v "$DV" -c "$MNAME2a" "$MNAME2b"
fi
