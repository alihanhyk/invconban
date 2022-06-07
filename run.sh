echo "cleaning data ..."
python src/data-clean.py

for key in {0..4}
do
    echo "generating semi-synthetic data for key=$key ..."
    python src/data.py --key $key
done

for agent in 0 1 11 12 2 21 3
do
    for key in {0..4}
    do

        echo "running main-birl.py for agent$agent-key$key ..."
        python src/main-birl.py -i "data/agent$agent-key$key.obj" -o "res/birl-agent$agent-key$key.obj"
        echo "running main-trex.py for agent$agent-key$key ..."
        python src/main-trex.py -i "data/agent$agent-key$key.obj" -o "res/trex-agent$agent-key$key.obj"
        echo "running main-ispi.py for agent$agent-key$key ..."
        python src/main-ispi.py -i "data/agent$agent-key$key.obj" -o "res/ispi-agent$agent-key$key.obj"

        echo "running main-irl-kfold.py with k=5 for agent$agent-key$key ..."
        python src/main-irl-kfold.py -k 5 -i "data/agent$agent-key$key.obj" -o "res/irl-kfold-k5-agent$agent-key$key.obj"
        echo "running main-irl-kfold.py with k=10 for agent$agent-key$key ..."
        python src/main-irl-kfold.py -k 10 -i "data/agent$agent-key$key.obj" -o "res/irl-kfold-k10-agent$agent-key$key.obj"
        echo "running main-irl-kfold.py with k=500 for agent$agent-key$key ..."
        python src/main-irl-kfold.py -k 500 -i "data/agent$agent-key$key.obj" -o "res/irl-kfold-k500-agent$agent-key$key.obj"

        echo "running main-ns-irl.py with k=5 for agent$agent-key$key ..."
        python src/main-ns-irl.py -k 5 -i "data/agent$agent-key$key.obj" -o "res/ns-irl-k5-agent$agent-key$key.obj"
        echo "running main-ns-irl.py with k=10 for agent$agent-key$key ..."
        python src/main-ns-irl.py -k 10 -i "data/agent$agent-key$key.obj" -o "res/ns-irl-k10-agent$agent-key$key.obj"

        echo "running main-bicb.py for agent$agent-key$key ..."
        python src/main-bicb.py -i "data/agent$agent-key$key.obj" -o "res/bicb-agent$agent-key$key.obj"
        echo "running main-nbicb.py for agent$agent-key$key ..."
        python src/main-nbicb.py -i "data/agent$agent-key$key.obj" -o "res/nbicb-agent$agent-key$key.obj"

    done
done
