#!/bin/sh
for indir in UD_French-ParTUT UD_French-GSD
do
  if   [ ! -d $indir ] ; then
      echo "cloning $indir data from github..."
      git clone https://github.com/UniversalDependencies/$indir.git
  else
      echo "$indir folder present. Nothing to be downloaded."
  fi

  INDIR=$indir/
  DATADIR="data/"

  if [ ! -d $DATADIR ] ; then
    echo "creating folder `data`"
    mkdir $DATADIR
  fi

  mkdir output
  echo "creating folder 'output'"
  mkdir model
  echo "creating folder 'model'"
  mkdir images
  echo "creating folder 'images'"

  for fil in fr_partut-ud-train.conllu fr_partut-ud-dev.conllu fr_partut-ud-test.conllu fr_gsd-ud-train.conllu fr_gsd-ud-dev.conllu fr_gsd-ud-test.conllu
  do  if [[ -f "$DATADIR$fil" ]] ; then
      echo "File $DATADIR/$fil present"
    else
      echo "File $DATADIR$fil absent, moving file."
      mv $INDIR$fil $DATADIR$fil
    fi
  done
done

exit