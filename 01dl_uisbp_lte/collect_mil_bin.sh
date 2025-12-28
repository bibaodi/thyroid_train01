echo Collect mil binary training output
RESDIR=$1
if [ -d "$RESDIR" ] 
then
    echo "Directory $RESDIR exists."
else
    mkdir $RESDIR
    echo "move files to $RESDIR"
    mv data $RESDIR
    mv mil_*.log $RESDIR
    mv validation_model_selection_binary.txt $RESDIR
fi

