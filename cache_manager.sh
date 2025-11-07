#!/bin/bash
# Cache Manager for Ramanlab Auto Data Analysis Pipeline
# Helps manage the state cache system to avoid redundant processing

CACHE_DIR="/home/ramanlab/Documents/cole/cache"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function show_usage() {
    echo "Usage: ./cache_manager.sh [command]"
    echo ""
    echo "Commands:"
    echo "  status        - Show current cache status"
    echo "  clear-all     - Clear all caches (forces full reprocessing)"
    echo "  clear-pipeline - Clear only pipeline caches"
    echo "  clear-combined - Clear only combined analysis cache"
    echo "  clear-reactions - Clear reaction prediction/matrix caches"
    echo "  clear-dataset <path> - Clear cache for specific dataset"
    echo "  list-datasets - List all cached datasets"
    echo ""
    echo "Examples:"
    echo "  ./cache_manager.sh status"
    echo "  ./cache_manager.sh clear-dataset /home/ramanlab/Documents/cole/Data/flys/opto_EB"
}

function show_status() {
    echo -e "${GREEN}=== Cache Status ===${NC}"
    echo ""

    if [ ! -d "$CACHE_DIR" ]; then
        echo -e "${YELLOW}Cache directory does not exist: $CACHE_DIR${NC}"
        return
    fi

    echo -e "${GREEN}Cache directory:${NC} $CACHE_DIR"
    echo -e "${GREEN}Total size:${NC} $(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)"
    echo ""

    # Pipeline caches
    if [ -d "$CACHE_DIR/pipeline" ]; then
        local count=$(find "$CACHE_DIR/pipeline" -name "state.json" | wc -l)
        echo -e "${GREEN}Pipeline caches:${NC} $count datasets"
    else
        echo -e "${YELLOW}Pipeline caches:${NC} None"
    fi

    # Combined cache
    if [ -f "$CACHE_DIR/combined/analysis/state.json" ]; then
        echo -e "${GREEN}Combined analysis cache:${NC} Present"
        local datasets=$(jq -r '.dataset_roots[]' "$CACHE_DIR/combined/analysis/state.json" 2>/dev/null | wc -l)
        echo "  └─ Covers $datasets datasets"
    else
        echo -e "${YELLOW}Combined analysis cache:${NC} None"
    fi

    # Reaction caches
    if [ -f "$CACHE_DIR/reaction_prediction/predict/state.json" ]; then
        echo -e "${GREEN}Reaction prediction cache:${NC} Present"
    else
        echo -e "${YELLOW}Reaction prediction cache:${NC} None"
    fi

    if [ -f "$CACHE_DIR/reaction_matrix/reaction_prediction/state.json" ]; then
        echo -e "${GREEN}Reaction matrix cache:${NC} Present"
    else
        echo -e "${YELLOW}Reaction matrix cache:${NC} None"
    fi

    echo ""
    echo -e "${GREEN}Next make run will:${NC}"

    # Check force flags in config.yaml
    local config="config.yaml"
    if [ -f "$config" ]; then
        local force_pipeline=$(grep -A 5 "^force:" "$config" | grep "pipeline:" | awk '{print $2}')
        local force_combined=$(grep -A 5 "^force:" "$config" | grep "combined:" | awk '{print $2}')

        if [ "$force_pipeline" = "true" ]; then
            echo -e "  ${RED}✗ Reprocess all pipelines (force.pipeline: true)${NC}"
        else
            echo -e "  ${GREEN}✓ Use cached pipelines (force.pipeline: false)${NC}"
        fi

        if [ "$force_combined" = "true" ]; then
            echo -e "  ${RED}✗ Reprocess combined analysis (force.combined: true)${NC}"
        else
            echo -e "  ${GREEN}✓ Use cached combined analysis (force.combined: false)${NC}"
        fi
    fi
}

function clear_all() {
    echo -e "${YELLOW}Clearing ALL caches...${NC}"
    if [ -d "$CACHE_DIR" ]; then
        rm -rf "$CACHE_DIR"/*
        echo -e "${GREEN}✓ All caches cleared${NC}"
        echo "Next 'make run' will perform full processing"
    else
        echo -e "${RED}Cache directory not found: $CACHE_DIR${NC}"
    fi
}

function clear_pipeline() {
    echo -e "${YELLOW}Clearing pipeline caches...${NC}"
    if [ -d "$CACHE_DIR/pipeline" ]; then
        rm -rf "$CACHE_DIR/pipeline"/*
        echo -e "${GREEN}✓ Pipeline caches cleared${NC}"
        echo "Next 'make run' will reprocess all datasets"
    else
        echo -e "${YELLOW}No pipeline caches to clear${NC}"
    fi
}

function clear_combined() {
    echo -e "${YELLOW}Clearing combined analysis cache...${NC}"
    if [ -d "$CACHE_DIR/combined" ]; then
        rm -rf "$CACHE_DIR/combined"/*
        echo -e "${GREEN}✓ Combined analysis cache cleared${NC}"
        echo "Next 'make run' will regenerate combined outputs"
    else
        echo -e "${YELLOW}No combined cache to clear${NC}"
    fi
}

function clear_reactions() {
    echo -e "${YELLOW}Clearing reaction prediction/matrix caches...${NC}"
    local cleared=false
    if [ -d "$CACHE_DIR/reaction_prediction" ]; then
        rm -rf "$CACHE_DIR/reaction_prediction"/*
        cleared=true
    fi
    if [ -d "$CACHE_DIR/reaction_matrix" ]; then
        rm -rf "$CACHE_DIR/reaction_matrix"/*
        cleared=true
    fi

    if [ "$cleared" = true ]; then
        echo -e "${GREEN}✓ Reaction caches cleared${NC}"
        echo "Next 'make run' will rerun predictions and matrices"
    else
        echo -e "${YELLOW}No reaction caches to clear${NC}"
    fi
}

function list_datasets() {
    echo -e "${GREEN}=== Cached Datasets ===${NC}"
    echo ""

    if [ ! -d "$CACHE_DIR/pipeline" ]; then
        echo -e "${YELLOW}No pipeline caches found${NC}"
        return
    fi

    local count=0
    for state_file in "$CACHE_DIR/pipeline"/*/state.json; do
        if [ -f "$state_file" ]; then
            local dir=$(dirname "$state_file")
            local key=$(basename "$dir")
            # Convert underscores back to path
            local path=$(echo "$key" | sed 's/_/\//g')
            count=$((count + 1))
            echo "$count. /$path"
        fi
    done

    if [ "$count" -eq 0 ]; then
        echo -e "${YELLOW}No cached datasets found${NC}"
    else
        echo ""
        echo -e "${GREEN}Total: $count datasets${NC}"
    fi
}

function clear_dataset() {
    local dataset_path="$1"

    if [ -z "$dataset_path" ]; then
        echo -e "${RED}Error: No dataset path provided${NC}"
        echo "Usage: ./cache_manager.sh clear-dataset <path>"
        return 1
    fi

    # Convert path to cache key format
    local clean_path=$(echo "$dataset_path" | sed 's/^\/*//')  # Remove leading slashes
    local cache_key=$(echo "$clean_path" | sed 's/\//_/g')    # Replace / with _

    local cache_path="$CACHE_DIR/pipeline/$cache_key"

    if [ -d "$cache_path" ]; then
        echo -e "${YELLOW}Clearing cache for dataset: $dataset_path${NC}"
        rm -rf "$cache_path"
        echo -e "${GREEN}✓ Cache cleared for $cache_key${NC}"
        echo "Next 'make run' will reprocess this dataset"
    else
        echo -e "${RED}No cache found for: $dataset_path${NC}"
        echo "Cache path checked: $cache_path"
        echo ""
        echo "Available datasets:"
        list_datasets
    fi
}

# Main command dispatcher
case "${1:-}" in
    status)
        show_status
        ;;
    clear-all)
        read -p "Clear ALL caches? This will force full reprocessing. (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            clear_all
        else
            echo "Cancelled"
        fi
        ;;
    clear-pipeline)
        clear_pipeline
        ;;
    clear-combined)
        clear_combined
        ;;
    clear-reactions)
        clear_reactions
        ;;
    clear-dataset)
        clear_dataset "$2"
        ;;
    list-datasets)
        list_datasets
        ;;
    *)
        show_usage
        ;;
esac
