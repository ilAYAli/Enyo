#! /bin/bash

resolve_path() {
    local directory="$1"
    local path="$2"
    if [[ "$path" =~ ^(/|\./|\.\./) ]]; then
        echo "$path"
    else
        echo "$HOME/assets/$directory/$path"
    fi
}

REFERENCE=reference
CANDIDATE=""
NET=candidate.net

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reference)
            REFERENCE="$2"
            shift 2
            ;;
        --candidate)
            CANDIDATE="$2"
            shift 2
            ;;
        --net)
            NET="$2"
            shift 2
            ;;
        *)
            echo "Error: Invalid argument '$1'. Only --candidate, --reference, and --net flags are allowed." >&2
            exit 2
            ;;
    esac
done

if [[ -z "$CANDIDATE" ]]; then
    echo "usage: $0 --candidate CANDIDATE_ENGINE [--reference REFERENCE_ENGINE] [--net NET]" >&2
    exit 2
fi

REFERENCE=$(resolve_path engines "$REFERENCE")
CANDIDATE=$(resolve_path engines "$CANDIDATE")
NET=$(resolve_path nets "$NET")
RUN="sprt-$(basename "$CANDIDATE")-vs-$(basename "$REFERENCE")-$(date +%Y%m%d-%H%M%S)"

set +x
forge sprt \
    --run "$RUN" \
    --wait \
    --comment "sprt $CANDIDATE vs $REFERENCE using $(basename "$NET")" \
    --book ~/assets/books/AntiDraw_V2.1/WOMP_Openings_V1/WOMP_V1_+150_+159/WOMP_V1_6mvs_big_+140_+169.epd \
    --candidate "$CANDIDATE" \
    --candidate-uci "nnue_file=$NET" \
    --concurrency 1 \
    --games 1000 \
    --reference "$REFERENCE" \
    --reference-uci "nnue_file=$NET" \
    --restart on \
    --shards 24 \
    --tc 10+0.1 \
    --threads 1

result=$(forge status "$RUN" --json)

elo=$(jq -r '.display.fields[] | select(startswith("elo=")) | split("=")[1]' <<< "$result")

llr=$(jq -r '.display.fields[] | select(startswith("llr=")) | split("=")[1] | split("/")[0]' <<< "$result")

printf 'elo=%s llr=%s\n' "$elo" "$llr"
