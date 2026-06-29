# amharic-finetuning

To translate a corpus, modify ```translate_corpus.sh``` to point at your data.

Then type ```sbatch translate_corpus.sh``` to start the process.

To check on the process, use:
```tail -f logs/logs_{ID}.err```
(keep in mind that if you don't have a logs directory, the process will crash)

The translated output will go into ```logs/logs_{ID}.out```

```squeue``` allows you to monitor the process queue.