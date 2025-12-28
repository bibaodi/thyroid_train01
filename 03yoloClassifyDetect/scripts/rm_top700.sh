for i in `head -n 700 ../../benign.list `; do CMD="rm $i" && echo "$CMD"; done

## Benign
### train
for i in `head -n 700 ../../benign.list `; do CMD="rm $i" && echo "$CMD" && eval $CMD; done
### val/test
for i in `tail -n 2786 ../../benign.list `; do CMD="rm $i" && echo "$CMD" && eval $CMD; done


## malign
### train
for i in `head -n 500 ../../malign.list `; do CMD="rm $i" && echo "$CMD" && eval $CMD; done
### val/test
for i in `tail -n 1952 ../../malign.list `; do CMD="rm $i" && echo "$CMD" && eval $CMD; done


