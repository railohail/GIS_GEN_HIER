"""
US States and Counties constants for hierarchical dataset generation.

This module provides mappings for:
- Level 0 (L0): US States - 50 classes (excluding DC/territories)
- Level 1 (L1): US Counties - ~3,200 classes

Shapefile columns used:
- States (ne_10m_admin_1_states_provinces):
  - name: Full state name (e.g., "California")
  - postal: 2-letter code (e.g., "CA")
  - fips: FIPS code with US prefix (e.g., "US06")

- Counties (ne_10m_admin_2_counties):
  - NAME: County name (e.g., "Los Angeles")
  - REGION: State abbreviation (e.g., "CA")
  - REGION_COD: State FIPS code (e.g., "06")
  - CODE_LOCAL: Full county FIPS (e.g., "06037")
"""

from typing import Final, Dict, List

# ============================================================================
# US State Mappings (50 states)
# ============================================================================

# State name to class ID (alphabetical order, 0-indexed)
STATE_TO_CLASS: Final[Dict[str, int]] = {
    'Alabama': 0,
    'Alaska': 1,
    'Arizona': 2,
    'Arkansas': 3,
    'California': 4,
    'Colorado': 5,
    'Connecticut': 6,
    'Delaware': 7,
    'Florida': 8,
    'Georgia': 9,
    'Hawaii': 10,
    'Idaho': 11,
    'Illinois': 12,
    'Indiana': 13,
    'Iowa': 14,
    'Kansas': 15,
    'Kentucky': 16,
    'Louisiana': 17,
    'Maine': 18,
    'Maryland': 19,
    'Massachusetts': 20,
    'Michigan': 21,
    'Minnesota': 22,
    'Mississippi': 23,
    'Missouri': 24,
    'Montana': 25,
    'Nebraska': 26,
    'Nevada': 27,
    'New Hampshire': 28,
    'New Jersey': 29,
    'New Mexico': 30,
    'New York': 31,
    'North Carolina': 32,
    'North Dakota': 33,
    'Ohio': 34,
    'Oklahoma': 35,
    'Oregon': 36,
    'Pennsylvania': 37,
    'Rhode Island': 38,
    'South Carolina': 39,
    'South Dakota': 40,
    'Tennessee': 41,
    'Texas': 42,
    'Utah': 43,
    'Vermont': 44,
    'Virginia': 45,
    'Washington': 46,
    'West Virginia': 47,
    'Wisconsin': 48,
    'Wyoming': 49,
}

# Reverse mapping: class ID to state name
CLASS_TO_STATE: Final[Dict[int, str]] = {v: k for k, v in STATE_TO_CLASS.items()}

# State postal code to full name
STATE_POSTAL_TO_NAME: Final[Dict[str, str]] = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming',
}

# State name to postal code
STATE_NAME_TO_POSTAL: Final[Dict[str, str]] = {v: k for k, v in STATE_POSTAL_TO_NAME.items()}

# State FIPS codes (2-digit, no "US" prefix)
STATE_FIPS_CODES: Final[Dict[str, str]] = {
    'Alabama': '01',
    'Alaska': '02',
    'Arizona': '04',
    'Arkansas': '05',
    'California': '06',
    'Colorado': '08',
    'Connecticut': '09',
    'Delaware': '10',
    'Florida': '12',
    'Georgia': '13',
    'Hawaii': '15',
    'Idaho': '16',
    'Illinois': '17',
    'Indiana': '18',
    'Iowa': '19',
    'Kansas': '20',
    'Kentucky': '21',
    'Louisiana': '22',
    'Maine': '23',
    'Maryland': '24',
    'Massachusetts': '25',
    'Michigan': '26',
    'Minnesota': '27',
    'Mississippi': '28',
    'Missouri': '29',
    'Montana': '30',
    'Nebraska': '31',
    'Nevada': '32',
    'New Hampshire': '33',
    'New Jersey': '34',
    'New Mexico': '35',
    'New York': '36',
    'North Carolina': '37',
    'North Dakota': '38',
    'Ohio': '39',
    'Oklahoma': '40',
    'Oregon': '41',
    'Pennsylvania': '42',
    'Rhode Island': '44',
    'South Carolina': '45',
    'South Dakota': '46',
    'Tennessee': '47',
    'Texas': '48',
    'Utah': '49',
    'Vermont': '50',
    'Virginia': '51',
    'Washington': '53',
    'West Virginia': '54',
    'Wisconsin': '55',
    'Wyoming': '56',
}

# Reverse: FIPS to state name
FIPS_TO_STATE: Final[Dict[str, str]] = {v: k for k, v in STATE_FIPS_CODES.items()}

# ============================================================================
# Shapefile Column Names (Natural Earth)
# ============================================================================

# States shapefile columns (ne_10m_admin_1_states_provinces)
STATE_SHAPEFILE_COLUMNS: Final[Dict[str, str]] = {
    'id_column': 'postal',       # 2-letter state code (WA, CA, etc.)
    'name_column': 'name',       # Full state name
    'name_en_column': 'name_en', # English name (same as name for US)
    'fips_column': 'fips',       # FIPS with US prefix (US53, US06, etc.)
    'filter_column': 'iso_a2',   # Use to filter US states only
    'filter_value': 'US',        # Value to filter for US
}

# Counties shapefile columns (ne_10m_admin_2_counties)
COUNTY_SHAPEFILE_COLUMNS: Final[Dict[str, str]] = {
    'id_column': 'CODE_LOCAL',   # Full county FIPS (state+county)
    'name_column': 'NAME',       # County name
    'name_en_column': 'NAME_EN', # English name
    'state_abbrev_column': 'REGION',      # State abbreviation (WA, CA)
    'state_fips_column': 'REGION_COD',    # State FIPS code (53, 06)
    'full_name_column': 'NAME_ALT',       # Full name with "County" suffix
}

# ============================================================================
# Test States (small subset for initial testing)
# ============================================================================

# States with fewer counties for quick testing
TEST_STATES_SMALL: Final[List[str]] = [
    'Delaware',     # 3 counties - smallest
    'Rhode Island', # 5 counties
    'Hawaii',       # 5 counties
]

# States with TIF files in Test_data/States_map/
TEST_STATES_WITH_TIFS: Final[List[str]] = [
    'Alabama',
    'Alaska',
    'Arizona',
    'California',
    'Florida',
    'Georgia',
]

# ============================================================================
# Data Directory Structure
# ============================================================================

# Expected directory structure for US data
US_DATA_STRUCTURE = """
Test_data/
  States_map/
    Alabama/
      *.tif           # Georeferenced map images
    Alaska/
      *.tif
    ...

ne_10m_admin_1_states_provinces/
  ne_10m_admin_1_states_provinces.shp  # State boundaries

ne_10m_admin_2_counties/
  ne_10m_admin_2_counties.shp          # County boundaries
"""
