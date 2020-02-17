cat embeddings/list_embeddings.txt | while read -ra line
do
    echo ${line[0]}
    echo ${line[1]}
    python3 main.py --reset_model --embedding_path "embeddings/${line[0]}" --log_dir "evaluations/${line[1]}/"
    python3 main.py --evaluate --test_set data/fr_gsd-ud-test.conllu --log_dir "evaluations/${line[1]}/"
    python3 main.py --evaluate --test_set data/fr_partut-ud-test.conllu --log_dir "evaluations/${line[1]}/"

    echo end
done
