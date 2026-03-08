#!/usr/bin/env python3
"""
Generate US administrative hierarchy JSON file.

Creates a mapping file from states (L0) to counties (L1) similar to
Taiwan's taiwan_admin_hierarchy.json.

Usage:
    python -m tools.generate_us_hierarchy
    python -m tools.generate_us_hierarchy --output us_admin_hierarchy.json
    python -m tools.generate_us_hierarchy --states Alabama California  # specific states only
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
from utils.core.us_constants import (
    STATE_TO_CLASS,
    STATE_NAME_TO_POSTAL,
    STATE_FIPS_CODES,
)


def generate_us_hierarchy(
    states_shapefile: str = 'ne_10m_admin_1_states_provinces',
    counties_shapefile: str = 'ne_10m_admin_2_counties',
    output_path: str = 'us_admin_hierarchy.json',
    state_filter: list = None,
):
    """
    Generate US state-county hierarchy JSON.

    Args:
        states_shapefile: Path to states shapefile (ne_10m_admin_1)
        counties_shapefile: Path to counties shapefile (ne_10m_admin_2)
        output_path: Where to save the JSON
        state_filter: Optional list of state names to include

    Returns:
        dict: The hierarchy data
    """
    print("Loading shapefiles...")

    # Load shapefiles
    states_gdf = gpd.read_file(states_shapefile)
    counties_gdf = gpd.read_file(counties_shapefile)

    # Filter to US states only
    us_states = states_gdf[states_gdf['iso_a2'] == 'US'].copy()
    print(f"Found {len(us_states)} US states/territories")

    # Filter to specific states if requested
    if state_filter:
        us_states = us_states[us_states['name'].isin(state_filter)]
        print(f"Filtered to {len(us_states)} requested states: {state_filter}")

    # Build hierarchy data
    hierarchy = {
        'metadata': {
            'country': 'United States',
            'l0_name': 'state',
            'l1_name': 'county',
            'l0_count': len(us_states),
            'l1_count': 0,  # Will be updated
            'source': 'Natural Earth 10m',
        },
        'states': {},
        'counties': {},
    }

    # Process each state
    county_class_id = 0

    for _, state_row in us_states.iterrows():
        state_name = state_row['name']
        state_postal = state_row['postal']

        # Skip if not in our mapping (DC, territories, etc.)
        if state_name not in STATE_TO_CLASS:
            print(f"  Skipping {state_name} (not in 50 states)")
            continue

        state_class_id = STATE_TO_CLASS[state_name]
        state_fips = STATE_FIPS_CODES.get(state_name, '')

        # Get counties for this state
        state_counties = counties_gdf[counties_gdf['REGION'] == state_postal].copy()

        print(f"  {state_name} ({state_postal}): {len(state_counties)} counties")

        # Add state entry
        hierarchy['states'][state_postal] = {
            'name': state_name,
            'english': state_name,
            'class_id': state_class_id,
            'fips': state_fips,
            'postal': state_postal,
            'county_count': len(state_counties),
            'county_ids': [],
        }

        # Process counties
        for _, county_row in state_counties.iterrows():
            county_name = county_row['NAME']
            county_fips = county_row['CODE_LOCAL']
            county_full_name = county_row.get('NAME_ALT', f"{county_name} County")

            # Create county ID (state_postal + sequential)
            county_id = f"{state_postal}{str(county_class_id).zfill(3)}"

            hierarchy['counties'][county_id] = {
                'name': county_name,
                'english': county_name,
                'full_name': county_full_name,
                'class_id': county_class_id,
                'fips': county_fips,
                'state_postal': state_postal,
                'state_name': state_name,
                'state_class_id': state_class_id,
            }

            hierarchy['states'][state_postal]['county_ids'].append(county_id)
            county_class_id += 1

    # Update metadata
    hierarchy['metadata']['l1_count'] = county_class_id

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)

    print(f"\nHierarchy saved to: {output_path}")
    print(f"  States (L0): {hierarchy['metadata']['l0_count']}")
    print(f"  Counties (L1): {hierarchy['metadata']['l1_count']}")

    return hierarchy


def main():
    parser = argparse.ArgumentParser(
        description="Generate US administrative hierarchy JSON"
    )
    parser.add_argument(
        '--states-shapefile',
        default='ne_10m_admin_1_states_provinces',
        help='Path to states shapefile'
    )
    parser.add_argument(
        '--counties-shapefile',
        default='ne_10m_admin_2_counties',
        help='Path to counties shapefile'
    )
    parser.add_argument(
        '--output', '-o',
        default='us_admin_hierarchy.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--states',
        nargs='+',
        help='Filter to specific states (e.g., --states Alabama California)'
    )

    args = parser.parse_args()

    generate_us_hierarchy(
        states_shapefile=args.states_shapefile,
        counties_shapefile=args.counties_shapefile,
        output_path=args.output,
        state_filter=args.states,
    )


if __name__ == '__main__':
    main()
