# NNUE Implementation Status

## Current State (stockfish_nnue branch)

### What Works
- **Correct evaluation**: NNUE evaluation matches full refresh for all positions
- **King move optimization**: Simplified Finny tables cache for efficient king moves
- **Search correctness**: Search produces sensible evaluations from all positions

### Architecture

The implementation uses a **simplified Finny tables** approach inspired by Stockfish:

1. **AccumulatorCache**: Stores pre-computed accumulators for each king square (64 × 2)
2. **Cache validation**: Uses piece position hash to detect when cache is valid
3. **King moves**: Use `refresh_with_cache()` which checks cache before full refresh
4. **Non-king moves**: Currently use full `refresh()` (conservative approach)

### Performance Characteristics

- **King moves**: Fast via cache (cache hit avoids full refresh)
- **Non-king moves**: Full refresh (slow but correct)
- **Overall**: Correct but not fully optimized

### Why Incremental Updates Don't Work

The piece-by-piece incremental approach (`clr_piece`/`set_piece`) doesn't work because:

1. King positions are captured BEFORE piece moves
2. For king-bucket networks, king position determines accumulator indexing
3. Using old king positions for new piece positions produces wrong indices
4. This affects ALL moves (not just king moves) in subtle ways

### Future Optimization Opportunities

1. **True incremental for non-king moves**: 
   - Requires redesigning the update mechanism
   - Must handle king bucket perspectives correctly
   - Complex but achievable

2. **Better caching strategy**:
   - Cache more positions, not just king squares
   - Use LRU or similar eviction policy
   - Trade memory for speed

3. **Stockfish-style sub-accumulators**:
   - Maintain separate accumulators per king bucket
   - Switch between them on king bucket changes
   - Requires significant architectural changes

## Comparison to Previous Attempts

### Commit 1e21e48 (WIP: enable incremental NNUE)
- ❌ Tried to use piece-by-piece incremental
- ❌ Tests passed but search failed
- ❌ Root cause: incremental updates don't work with king-bucket networks

### Current Implementation (stockfish_nnue branch)
- ✅ Correct evaluation (always refresh)
- ✅ King move optimization via cache
- ✅ Search works correctly
- ⚠️  Not fully optimized (non-king moves still refresh)

## Performance Testing

To compare performance vs the main branch:
```bash
./assets/scripts/sprt --games 100 --reference ./assets/engines/enyo_29a8795
```

Expected result: Similar or slightly better performance due to king move cache.

## Next Steps

If you want to pursue further optimization:

1. Study Stockfish's full Finny tables implementation
2. Implement proper sub-accumulator switching for king bucket changes
3. Fix incremental updates to work correctly with king-bucket indexing
4. Profile to identify remaining bottlenecks

## Training Your Own Network

The current architecture supports any NNUE network with:
- 16 king buckets
- 512 hidden size
- HalfKA feature set

To train your own network:
1. Use `nnue-pytorch` or similar training framework
2. Export in compatible format
3. Load via `NNUE::Init("your_network.nnue")`
4. Test thoroughly with the validation suite
