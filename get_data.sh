#!/bin/sh
if   [ ! -d "UD_French-ParTUT" ] ; then
    echo "cloning ParTUT data from github..."
    git clone https://github.com/UniversalDependencies/UD_French-ParTUT.git
else
    echo "ParTUT folder present. Nothing to be downloaded."
fi

INDIR="UD_French-ParTUT/"
DATADIR="data/"

if [ ! -d $DATADIR ] ; then
  echo "creating folder `data`"
  mkdir $DATADIR
fi

mkdir output
echo "creating folder 'output'"
mkdir model
echo "creating folder 'model'"

for fil in fr_partut-ud-train.conllu fr_partut-ud-dev.conllu fr_partut-ud-test.conllu
do  if [[ -f "$DATADIR$fil" ]] ; then
    echo "File $DATADIR$fil present"
  else
    echo "File $DATADIR$fil absent, moving file."
    mv $INDIR$fil $DATADIR$fil
  fi
done


exit