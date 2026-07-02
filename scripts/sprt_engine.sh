#! /bin/bash

if (( $# != 2 )); then
    echo "usage: $0 REFERENCE_ENGINE CANDIDATE_ENGINE" >&2
    exit 2
fi

REFERENCE=$1
CANDIDATE=$2
RUN="sprt-$(basename "$CANDIDATE")-vs-$(basename "$REFERENCE")-$(date +%Y%m%d-%H%M%S)"

set +x
forge sprt \
    --run "$RUN" \
    --wait \
    --comment "sprt $CANDIDATE vs $REFERENCE" \
    --book ~/assets/books/AntiDraw_V2.1/WOMP_Openings_V1/WOMP_V1_+150_+159/WOMP_V1_6mvs_big_+140_+169.epd \
    --candidate "$CANDIDATE" \
    --candidate-uci nnue_file=~/assets/nets/default.net \
    --concurrency 1 \
    --games 1000 \
    --reference "$REFERENCE" \
    --reference-uci nnue_file=~/assets/nets/default.net \
    --restart on \
    --shards 24 \
    --tc 10+0.1 \
    --threads 1

result=$(forge status "$RUN" --json)

elo=$(jq -r '.display.fields[] | select(startswith("elo=")) | split("=")[1]' <<< "$result")

llr=$(jq -r '.display.fields[] | select(startswith("llr=")) | split("=")[1] | split("/")[0]' <<< "$result")

printf 'elo=%s llr=%s\n' "$elo" "$llr"
