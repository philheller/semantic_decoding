echo "Running tests for Decoding methods."

python greedy_search_test.py
python beam_search_test.py
python group_beam_search_test.py
python contrastive_search_test.py

echo "All tests performed."