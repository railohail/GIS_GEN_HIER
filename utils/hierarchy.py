"""
Hierarchical annotation utilities for H-DETR dataset generation.

This module handles:
- Loading and managing administrative hierarchies (Taiwan or US)
- Mapping between shapefile IDs and class IDs for both levels
- Parent-child relationship lookups
- Hierarchical YOLO annotation writing

Supported Regions:
- Taiwan:
  - Level 0 (L0): County (縣市) - 22 classes
  - Level 1 (L1): Township (鄉鎮區) - 368 classes

- United States:
  - Level 0 (L0): State - 50 classes
  - Level 1 (L1): County - ~3,200 classes
"""

import json
import os
import geopandas as gpd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class HierarchyLevel:
    """Represents one level in the administrative hierarchy."""
    name: str
    num_classes: int
    shapefile_path: str
    id_column: str
    name_column: str
    parent_id_column: Optional[str] = None  # Only for L1+

    # Filter options (for US data - filter states by iso_a2='US')
    filter_column: Optional[str] = None
    filter_value: Optional[str] = None

    # For L1: column containing state abbreviation for filtering
    state_abbrev_column: Optional[str] = None

    # Loaded data (populated by load())
    gdf: Optional[gpd.GeoDataFrame] = field(default=None, repr=False)
    id_to_class: Dict[str, int] = field(default_factory=dict)
    class_to_id: Dict[int, str] = field(default_factory=dict)
    id_to_name: Dict[str, str] = field(default_factory=dict)

    def load(self, state_filter: Optional[List[str]] = None) -> bool:
        """
        Load the shapefile and build ID mappings.

        Args:
            state_filter: Optional list of state abbreviations to filter L1 data
                         (e.g., ['AL', 'CA'] for Alabama and California counties)
        """
        if not os.path.exists(self.shapefile_path):
            print(f"ERROR: Shapefile not found: {self.shapefile_path}")
            return False

        self.gdf = gpd.read_file(self.shapefile_path)

        # Apply filter if specified (e.g., filter to US states only)
        if self.filter_column and self.filter_value:
            original_count = len(self.gdf)
            self.gdf = self.gdf[self.gdf[self.filter_column] == self.filter_value].copy()
            print(f"  Filtered {self.name}: {original_count} -> {len(self.gdf)} (by {self.filter_column}={self.filter_value})")

        # Apply state filter for L1 (counties)
        if state_filter and self.state_abbrev_column:
            original_count = len(self.gdf)
            self.gdf = self.gdf[self.gdf[self.state_abbrev_column].isin(state_filter)].copy()
            print(f"  Filtered {self.name} to states {state_filter}: {original_count} -> {len(self.gdf)}")

        # Reset index after filtering
        self.gdf = self.gdf.reset_index(drop=True)

        # Build mappings
        for idx, row in self.gdf.iterrows():
            entity_id = str(row[self.id_column])
            entity_name = row[self.name_column]
            class_id = idx  # Use row index as class ID

            self.id_to_class[entity_id] = class_id
            self.class_to_id[class_id] = entity_id
            self.id_to_name[entity_id] = entity_name

        print(f"Loaded {self.name} level: {len(self.gdf)} entities")
        return True

    def get_class_id(self, entity_id: str) -> Optional[int]:
        """Get class ID for an entity ID."""
        return self.id_to_class.get(entity_id)

    def get_entity_id(self, class_id: int) -> Optional[str]:
        """Get entity ID for a class ID."""
        return self.class_to_id.get(class_id)

    def get_name(self, entity_id: str) -> Optional[str]:
        """Get entity name for an entity ID."""
        return self.id_to_name.get(entity_id)


class HierarchyManager:
    """
    Manages the administrative hierarchy for hierarchical dataset generation.

    Supports both Taiwan and US administrative hierarchies:
    - Taiwan: County (L0) → Township (L1)
    - US: State (L0) → County (L1)

    Usage:
        manager = HierarchyManager()
        manager.load_from_config(config)

        # Get class IDs
        county_class = manager.get_county_class_id('A')  # Taiwan: 臺北市
        township_class = manager.get_township_class_id('A01')  # Taiwan: 中正區

        # For US:
        state_class = manager.get_county_class_id('AL')  # Alabama
        county_class = manager.get_township_class_id('01001')  # Autauga County

        # Get parent relationship
        parent_county_id = manager.get_parent_county_id('A01')  # Returns 'A'
        parent_county_class = manager.get_parent_county_class('A01')  # Returns class ID
    """

    def __init__(self):
        self.county_level: Optional[HierarchyLevel] = None  # L0: county (TW) or state (US)
        self.township_level: Optional[HierarchyLevel] = None  # L1: township (TW) or county (US)
        self.hierarchy_data: Dict[str, Any] = {}
        self.loaded = False

        # Region type: 'taiwan' or 'us'
        self.region_type: str = 'taiwan'

        # Township → County parent mapping
        self.township_to_parent: Dict[str, str] = {}  # township_id → county_id

        # For US: state abbreviation to postal code mapping
        self.state_abbrev_to_id: Dict[str, str] = {}

    def load_from_config(self, config: Dict[str, Any]) -> bool:
        """
        Load hierarchy from configuration dictionary.

        Args:
            config: Configuration dict with 'hierarchy' section

        Returns:
            True if loaded successfully
        """
        hierarchy_config = config.get('hierarchy', {})

        if not hierarchy_config.get('enabled', False):
            print("Hierarchical mode disabled in config")
            return False

        # Determine region type
        self.region_type = hierarchy_config.get('region_type', 'taiwan')
        print(f"Loading hierarchy for region: {self.region_type.upper()}")

        # Get districts to determine state filter for US
        districts = config.get('districts', [])
        state_filter = None

        if self.region_type == 'us' and districts:
            # Convert district names to state abbreviations for filtering
            from .core.us_constants import STATE_NAME_TO_POSTAL
            state_filter = []
            for district in districts:
                # District name should match state name (e.g., "Alabama")
                postal = STATE_NAME_TO_POSTAL.get(district)
                if postal:
                    state_filter.append(postal)
                else:
                    print(f"  Warning: Unknown state '{district}'")
            print(f"  State filter: {state_filter}")

        # Load Level 0 (County for Taiwan, State for US)
        l0_config = hierarchy_config.get('level_0', {})
        self.county_level = HierarchyLevel(
            name=l0_config.get('name', 'county'),
            num_classes=l0_config.get('num_classes', 22),
            shapefile_path=l0_config.get('shapefile', ''),
            id_column=l0_config.get('id_column', 'COUNTYID'),
            name_column=l0_config.get('name_column', 'COUNTYNAME'),
            filter_column=l0_config.get('filter_column'),
            filter_value=l0_config.get('filter_value'),
        )

        if not self.county_level.load():
            return False

        # For US, filter L0 to only requested states
        if self.region_type == 'us' and state_filter:
            id_col = self.county_level.id_column
            original_count = len(self.county_level.gdf)
            self.county_level.gdf = self.county_level.gdf[
                self.county_level.gdf[id_col].isin(state_filter)
            ].reset_index(drop=True)

            # Rebuild mappings after filter
            self.county_level.id_to_class.clear()
            self.county_level.class_to_id.clear()
            self.county_level.id_to_name.clear()

            for idx, row in self.county_level.gdf.iterrows():
                entity_id = str(row[id_col])
                entity_name = row[self.county_level.name_column]
                self.county_level.id_to_class[entity_id] = idx
                self.county_level.class_to_id[idx] = entity_id
                self.county_level.id_to_name[entity_id] = entity_name

            print(f"  Filtered L0 to {state_filter}: {original_count} -> {len(self.county_level.gdf)}")

        # Load Level 1 (Township for Taiwan, County for US)
        l1_config = hierarchy_config.get('level_1', {})
        self.township_level = HierarchyLevel(
            name=l1_config.get('name', 'township'),
            num_classes=l1_config.get('num_classes', 368),
            shapefile_path=l1_config.get('shapefile', ''),
            id_column=l1_config.get('id_column', 'TOWNID'),
            name_column=l1_config.get('name_column', 'TOWNNAME'),
            parent_id_column=l1_config.get('parent_id_column', 'COUNTYID'),
            state_abbrev_column=l1_config.get('state_abbrev_column'),
        )

        if not self.township_level.load(state_filter=state_filter):
            return False

        # Build parent-child mappings
        self._build_parent_mappings()

        # Load hierarchy JSON if available
        hierarchy_file = hierarchy_config.get('hierarchy_file', '')
        if hierarchy_file and os.path.exists(hierarchy_file):
            with open(hierarchy_file, 'r', encoding='utf-8') as f:
                self.hierarchy_data = json.load(f)
            print(f"Loaded hierarchy data from: {hierarchy_file}")

        self.loaded = True
        return True

    def _build_parent_mappings(self):
        """Build township → county parent mappings from shapefile."""
        if self.township_level.gdf is None:
            return

        parent_col = self.township_level.parent_id_column
        id_col = self.township_level.id_column

        # For US data, we need to map REGION_COD (state FIPS) to state postal code
        # The counties use REGION_COD (e.g., "01") and states use postal (e.g., "AL")
        if self.region_type == 'us':
            # Build FIPS to postal mapping from states
            fips_to_postal = {}
            if self.county_level.gdf is not None:
                for _, row in self.county_level.gdf.iterrows():
                    postal = str(row[self.county_level.id_column])
                    # Get FIPS from the fips column (e.g., "US01" -> "01")
                    fips = row.get('fips', '')
                    if fips and fips.startswith('US'):
                        fips = fips[2:]  # Remove "US" prefix
                    if fips:
                        fips_to_postal[fips] = postal

            # Also use state_abbrev_column if available
            state_abbrev_col = self.township_level.state_abbrev_column

            for _, row in self.township_level.gdf.iterrows():
                township_id = str(row[id_col])

                # Try state_abbrev_column first (REGION = "AL")
                if state_abbrev_col and state_abbrev_col in row.index:
                    parent_id = str(row[state_abbrev_col])
                elif parent_col and parent_col in row.index:
                    # parent_col might be REGION_COD (FIPS "01")
                    parent_fips = str(row[parent_col])
                    parent_id = fips_to_postal.get(parent_fips, parent_fips)
                else:
                    parent_id = None

                if parent_id:
                    self.township_to_parent[township_id] = parent_id
        else:
            # Taiwan: direct mapping
            for _, row in self.township_level.gdf.iterrows():
                township_id = str(row[id_col])
                parent_id = str(row[parent_col])
                self.township_to_parent[township_id] = parent_id

        print(f"Built parent mappings for {len(self.township_to_parent)} {self.township_level.name}s")

    def get_county_class_id(self, county_id: str) -> Optional[int]:
        """Get class ID for a county."""
        if self.county_level:
            return self.county_level.get_class_id(county_id)
        return None

    def get_township_class_id(self, township_id: str) -> Optional[int]:
        """Get class ID for a township."""
        if self.township_level:
            return self.township_level.get_class_id(township_id)
        return None

    def get_parent_county_id(self, township_id: str) -> Optional[str]:
        """Get parent county ID for a township."""
        return self.township_to_parent.get(township_id)

    def get_parent_county_class(self, township_id: str) -> Optional[int]:
        """Get parent county class ID for a township."""
        parent_id = self.get_parent_county_id(township_id)
        if parent_id:
            return self.get_county_class_id(parent_id)
        return None

    def get_county_gdf(self) -> Optional[gpd.GeoDataFrame]:
        """Get county GeoDataFrame."""
        if self.county_level:
            return self.county_level.gdf
        return None

    def get_township_gdf(self) -> Optional[gpd.GeoDataFrame]:
        """Get township GeoDataFrame."""
        if self.township_level:
            return self.township_level.gdf
        return None

    def get_townships_in_county(self, county_id: str) -> List[str]:
        """Get all township IDs within a county."""
        return [
            tid for tid, cid in self.township_to_parent.items()
            if cid == county_id
        ]

    def get_county_id_by_district_name(self, district_name: str) -> Optional[str]:
        """
        Get county ID from district name (e.g., 'miaoli' -> 'K').

        Searches hierarchy_data for matching English name.

        Args:
            district_name: District name (e.g., 'miaoli', 'taipei', 'new_taipei')

        Returns:
            County ID (e.g., 'K') or None if not found
        """
        if not self.hierarchy_data or 'counties' not in self.hierarchy_data:
            return None

        # Normalize input
        search_name = district_name.lower().replace('_', ' ')

        for county_id, info in self.hierarchy_data['counties'].items():
            english = info.get('english', '').lower()
            # Check if district name is part of English name
            # e.g., 'miaoli' in 'miaoli county', 'new taipei' in 'new taipei city'
            if search_name in english.replace('_', ' '):
                return county_id

        return None

    def build_township_adjacency_graph(self, county_id: str) -> Dict[str, set]:
        """
        Build adjacency graph for townships within a county.

        Uses Shapely's .touches() method to determine adjacency.

        Args:
            county_id: County ID to build graph for

        Returns:
            Dict mapping township_id -> set of adjacent township_ids
        """
        adjacency: Dict[str, set] = {}
        townships = self.get_townships_in_county(county_id)

        if self.township_level is None or self.township_level.gdf is None:
            return adjacency

        id_col = self.township_level.id_column

        # Filter to only townships in this county
        township_gdf = self.township_level.gdf[
            self.township_level.gdf[id_col].isin(townships)
        ]

        # Build adjacency using geometry touches
        for idx1, row1 in township_gdf.iterrows():
            tid1 = row1[id_col]
            adjacency[tid1] = set()

            for idx2, row2 in township_gdf.iterrows():
                if idx1 == idx2:
                    continue
                tid2 = row2[id_col]

                # Check if geometries touch (share boundary but don't overlap)
                # Also check intersects for robustness
                if row1.geometry.touches(row2.geometry) or \
                   (row1.geometry.intersects(row2.geometry) and not row1.geometry.equals(row2.geometry)):
                    adjacency[tid1].add(tid2)

        return adjacency

    def get_edge_townships(self, county_id: str) -> List[str]:
        """
        Get townships that are on the edge of a county.

        Edge townships are those whose geometry touches the county boundary
        (i.e., they have exterior edges not shared with other townships).

        Args:
            county_id: County ID

        Returns:
            List of township IDs on county edge
        """
        edge_townships = []

        if self.county_level is None or self.county_level.gdf is None:
            return edge_townships
        if self.township_level is None or self.township_level.gdf is None:
            return edge_townships

        # Get county geometry
        county_gdf = self.county_level.gdf
        county_id_col = self.county_level.id_column
        county_row = county_gdf[county_gdf[county_id_col] == county_id]

        if county_row.empty:
            return edge_townships

        county_geom = county_row.iloc[0].geometry
        county_boundary = county_geom.boundary

        # Get townships in this county
        townships = self.get_townships_in_county(county_id)
        township_id_col = self.township_level.id_column
        township_gdf = self.township_level.gdf[
            self.township_level.gdf[township_id_col].isin(townships)
        ]

        for _, row in township_gdf.iterrows():
            township_id = row[township_id_col]
            township_boundary = row.geometry.boundary

            # Check if township boundary intersects county boundary
            # This means the township is on the edge of the county
            if township_boundary.intersects(county_boundary):
                edge_townships.append(township_id)

        return edge_townships

    def select_dropout_townships(
        self,
        county_id: str,
        mode: str,
        config: Dict[str, Any],
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Select townships to drop based on mode and configuration.

        Args:
            county_id: County ID
            mode: 'random' or 'config'
            config: Dropout configuration dict
            seed: Random seed for reproducibility

        Returns:
            List of township IDs to drop
        """
        import random

        if mode == 'config':
            # Look up explicit dropouts
            dropouts = config.get('config', {}).get('dropouts', [])
            for dropout in dropouts:
                if dropout.get('county_id') == county_id:
                    return dropout.get('township_ids', [])
            return []

        elif mode == 'random':
            random_config = config.get('random', {})
            # Handle both dict and dataclass-style access
            if hasattr(random_config, 'min_drop'):
                min_drop = random_config.min_drop
                max_drop = random_config.max_drop
                edge_only = random_config.edge_only
                cluster_drop = random_config.cluster_drop
            else:
                min_drop = random_config.get('min_drop', 1)
                max_drop = random_config.get('max_drop', 3)
                edge_only = random_config.get('edge_only', True)
                cluster_drop = random_config.get('cluster_drop', True)

            if seed is not None:
                random.seed(seed)

            # Get candidate townships
            if edge_only:
                candidates = self.get_edge_townships(county_id)
            else:
                candidates = self.get_townships_in_county(county_id)

            if not candidates:
                return []

            # Determine number to drop
            num_drop = random.randint(min_drop, min(max_drop, len(candidates)))

            if cluster_drop and num_drop > 1:
                # Use random walk to select adjacent townships
                return self._select_adjacent_cluster(county_id, candidates, num_drop)
            else:
                # Simple random selection
                return random.sample(candidates, num_drop)

        return []

    def _select_adjacent_cluster(
        self,
        county_id: str,
        candidates: List[str],
        num_select: int
    ) -> List[str]:
        """
        Select a cluster of adjacent townships using random walk.

        Starts from a random edge township and grows outward,
        ensuring selected townships form a contiguous group.

        Args:
            county_id: County ID
            candidates: List of candidate township IDs
            num_select: Number of townships to select

        Returns:
            List of adjacent township IDs
        """
        import random

        adjacency = self.build_township_adjacency_graph(county_id)

        if not candidates:
            return []

        # Start from random candidate
        selected = [random.choice(candidates)]

        while len(selected) < num_select:
            # Find all neighbors of selected townships
            neighbors = set()
            for tid in selected:
                neighbors.update(adjacency.get(tid, set()))

            # Remove already selected
            neighbors -= set(selected)

            if not neighbors:
                break  # No more neighbors available

            # Prioritize candidates (edge townships) if available
            candidate_neighbors = neighbors & set(candidates)
            if candidate_neighbors:
                next_township = random.choice(list(candidate_neighbors))
            else:
                next_township = random.choice(list(neighbors))

            selected.append(next_township)

        return selected

    def create_county_categories(self) -> List[Dict[str, Any]]:
        """Create COCO-style categories for L0 (counties/states) with English names."""
        if not self.county_level or self.county_level.gdf is None:
            return []

        # Build English name lookup from hierarchy_data
        english_names = {}

        if self.region_type == 'us':
            # US: Look up from 'states' key in hierarchy_data
            if self.hierarchy_data and 'states' in self.hierarchy_data:
                for state_code, state_info in self.hierarchy_data['states'].items():
                    english_names[state_code] = state_info.get('english', state_info.get('name', ''))
        else:
            # Taiwan: Look up from 'counties' key
            if self.hierarchy_data and 'counties' in self.hierarchy_data:
                for county_code, county_info in self.hierarchy_data['counties'].items():
                    english_names[county_code] = county_info.get('english', '')

        categories = []
        for class_id, entity_id in self.county_level.class_to_id.items():
            name = self.county_level.id_to_name.get(entity_id, f"{self.county_level.name}_{class_id}")

            if self.region_type == 'us':
                # US: entity_id is postal code (e.g., "AL")
                english_name = english_names.get(entity_id, name)
                supercategory = 'state'
            else:
                # Taiwan: first char of entity_id is usually the county code
                county_code = entity_id[0] if entity_id else ''
                english_name = english_names.get(county_code, f"County_{class_id}")
                supercategory = 'county'

            categories.append({
                'id': class_id,
                'name': name,
                'english': english_name,
                'supercategory': supercategory,
                'entity_id': entity_id,
            })
        return sorted(categories, key=lambda x: x['id'])

    def create_township_categories(self) -> List[Dict[str, Any]]:
        """Create COCO-style categories for L1 (townships/counties) with English names."""
        if not self.township_level or self.township_level.gdf is None:
            return []

        # Build English name lookup from hierarchy_data
        english_names = {}

        if self.region_type == 'us':
            # US: Look up from 'counties' key in hierarchy_data
            if self.hierarchy_data and 'counties' in self.hierarchy_data:
                for county_id, county_info in self.hierarchy_data['counties'].items():
                    # Map by FIPS code
                    fips = county_info.get('fips', '')
                    if fips:
                        english_names[fips] = county_info.get('english', county_info.get('name', ''))
        else:
            # Taiwan: Look up from 'townships' key
            if self.hierarchy_data and 'townships' in self.hierarchy_data:
                for town_id, town_info in self.hierarchy_data['townships'].items():
                    english_names[town_id] = town_info.get('english', '')

        categories = []
        for class_id, entity_id in self.township_level.class_to_id.items():
            name = self.township_level.id_to_name.get(entity_id, f"{self.township_level.name}_{class_id}")
            parent_id = self.get_parent_county_id(entity_id)
            parent_class = self.get_parent_county_class(entity_id)

            if self.region_type == 'us':
                # US: entity_id is FIPS code, name is already in English
                english_name = english_names.get(entity_id, name)
                supercategory = 'county'
            else:
                # Taiwan: entity_id is township code
                english_name = english_names.get(entity_id, f"T{class_id}")
                supercategory = 'township'

            categories.append({
                'id': class_id,
                'name': name,
                'english': english_name,
                'supercategory': supercategory,
                'entity_id': entity_id,
                'parent_id': parent_id,
                'parent_class': parent_class,
            })
        return sorted(categories, key=lambda x: x['id'])


# Global hierarchy manager instance
_hierarchy_manager: Optional[HierarchyManager] = None


def get_hierarchy_manager() -> HierarchyManager:
    """Get or create the global hierarchy manager."""
    global _hierarchy_manager
    if _hierarchy_manager is None:
        _hierarchy_manager = HierarchyManager()
    return _hierarchy_manager


def initialize_hierarchy(config: Dict[str, Any]) -> bool:
    """
    Initialize the hierarchy manager from config.

    Args:
        config: Configuration dictionary

    Returns:
        True if hierarchical mode is enabled and loaded successfully
    """
    manager = get_hierarchy_manager()
    return manager.load_from_config(config)


def is_hierarchical_mode(config: Dict[str, Any]) -> bool:
    """Check if hierarchical mode is enabled in config."""
    return config.get('hierarchy', {}).get('enabled', False)
