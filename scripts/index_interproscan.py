import json
import gzip
import numpy as np
import os
import io
import threading
from google.cloud import storage
from tqdm import tqdm
from collections import defaultdict

# Lock for thread-safe file operations
lock = threading.Lock()

class GOTermsExtractor:
    """
    Extracts GO terms from JSON.GZ files across multiple folders, 
    writing the data to a text file with position indexes and creating a 
    mapping from UniRef IDs to sequence indices.
    """
    
    def __init__(self, bucket_name, base_prefix, start_folder=0, end_folder=704,
                 output_dir="interproscan_data", index_dir="interproscan_idx"):
        """
        Initialize the GO terms extractor.
        
        Args:
            bucket_name: GCS bucket name
            base_prefix: Base prefix for folders (e.g., "output3/folder_")
            start_folder: Starting folder number (inclusive)
            end_folder: Ending folder number (inclusive)
            output_dir: Directory to store text output files
            index_dir: Directory to store index files
        """
        self.bucket_name = bucket_name
        self.base_prefix = base_prefix
        self.start_folder = start_folder
        self.end_folder = end_folder
        self.output_dir = output_dir
        self.index_dir = index_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)
        
        # Set up GCS client
        self.client = storage.Client(project='esm-multimer')
        self.bucket = self.client.bucket(bucket_name)
    
    def get_folder_prefix(self, folder_num):
        """Generate folder prefix with proper zero-padding"""
        return f"{self.base_prefix}{folder_num:03d}"
    
    def list_folder_files(self, folder_num):
        """List all JSON.GZ files in a specific folder"""
        prefix = self.get_folder_prefix(folder_num)
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        json_gz_blobs = [blob for blob in blobs if blob.name.endswith('.json.gz')]
        print(f"Found {len(json_gz_blobs)} JSON.GZ files in gs://{self.bucket_name}/{prefix}")
        return json_gz_blobs
    
    def extract_go_terms(self, data):
        """
        Extract GO terms and their locations from the JSON data.
        
        Args:
            data: JSON data containing InterProScan results
            
        Returns:
            Dictionary with sequence_id as key and GO terms information as value
        """
        results = {}
        
        # Process each result
        for result in data.get('results', []):
            # Skip if no matches or no sequence
            if not result.get('matches') or not result.get('sequence'):
                continue
                
            # Get sequence and ID (use first xref or a default if not available)
            sequence = result.get('sequence', '')
            
            # Extract sequence_id from xref (if available)
            sequence_id = None
            if result.get('xref') and len(result['xref']) > 0:
                sequence_id = result['xref'][0].get('id')
            
            # If no sequence_id found, try alternative sources or skip
            if not sequence_id:
                # Try to get ID from other fields or generate a placeholder
                sequence_id = result.get('id', f"unknown_{hash(sequence) % 10000000}")
                if not sequence_id:
                    continue
            
            go_terms_locations = []  # List to store GO terms and their locations

            # Process each match
            for match in result['matches']:
                # Skip if no locations
                if not match.get('locations'):
                    continue

                # Get locations
                locations = [(loc.get('start', 0), loc.get('end', 0)) 
                             for loc in match.get('locations', [])
                             if loc.get('start') is not None and loc.get('end') is not None]

                # Skip if no valid locations
                if not locations:
                    continue

                # CHANGE 1: Extract GO terms from signature (original method)
                if match.get('signature') and match['signature'].get('entry') and match['signature']['entry'].get('goXRefs'):
                    for go_ref in match['signature']['entry'].get('goXRefs', []):
                        if not go_ref.get('id'):
                            continue

                        go_term = {
                            'name': go_ref.get('name', ''),
                            'id': go_ref.get('id', '')
                        }

                        # Add (go_term, locations) pair
                        go_terms_locations.append({
                            'go_term': go_term,
                            'locations': locations
                        })

                # CHANGE 2: Extract GO terms directly from match (new method)
                if match.get('goXRefs'):
                    for go_ref in match['goXRefs']:
                        if not go_ref.get('id'):
                            continue

                        go_term = {
                            'name': go_ref.get('name', ''),
                            'id': go_ref.get('id', '')
                        }

                        # Add (go_term, locations) pair
                        go_terms_locations.append({
                            'go_term': go_term,
                            'locations': locations
                        })

            # Only add sequences that have GO terms
            if go_terms_locations:
                results[sequence_id] = {
                    'sequence': sequence,
                    'go_annotations': go_terms_locations
                }

        return results
    
    def format_go_entry(self, sequence_id, sequence, go_annotations):
        """
        Format a GO term entry according to the specified format.
        
        Args:
            sequence_id: UniRef sequence ID
            sequence: The actual sequence
            go_annotations: List of GO annotations with their locations
            
        Returns:
            Formatted string in the specified format
        """
        lines = [f">{sequence_id}", sequence]
        
        # Add GO terms with their locations
        for annotation in go_annotations:
            go_id = annotation['go_term']['id']
            
            # Add each location for this GO term
            for start, end in annotation['locations']:
                lines.append(f"{go_id} {start} {end}")

        # Join with newlines
        return '\n'.join(lines)

    def process_all_folders(self, go_terms_file="go_terms.txt", positions_file="go_positions.idx", 
                          id_to_index_file="uniref_to_index.txt"):
        """
        Process all folders in the specified range, extracting GO terms and writing them
        to a text file with appropriate indexing.
        
        Args:
            go_terms_file: Name of the text file to store GO terms data
            positions_file: Name of the file to store position indexes
            id_to_index_file: Name of the file to store UniRef ID to index mapping
            
        Returns:
            Number of sequences processed
        """
        go_terms_path = os.path.join(self.output_dir, go_terms_file)
        positions_path = os.path.join(self.index_dir, positions_file)
        id_to_index_path = os.path.join(self.index_dir, id_to_index_file)
        
        # For tracking positions
        positions = []
        current_position = 0
        
        # For mapping UniRef IDs to indices
        id_to_index = {}
        sequence_index = 0
        
        # Open the text file for streaming writes
        with open(go_terms_path, 'w') as outfile:
            # Process each folder
            for folder_num in range(self.start_folder, self.end_folder + 1):
                print(f"\nProcessing folder {folder_num:03d}...")
                
                # Get all JSON.GZ files in this folder
                json_gz_blobs = self.list_folder_files(folder_num)
                
                # Process each file in the folder
                for blob in tqdm(json_gz_blobs, desc=f"Folder {folder_num:03d}"):
                    try:
                        # Download content to memory
                        compressed_content = blob.download_as_bytes()
                        
                        # Decompress the content
                        with gzip.GzipFile(fileobj=io.BytesIO(compressed_content), mode='rb') as f:
                            # Load JSON data
                            json_content = json.loads(f.read().decode('utf-8'))
                        
                        # Extract GO terms
                        go_terms_data = self.extract_go_terms(json_content)
                        
                        # Process each sequence with GO terms
                        for sequence_id, data in go_terms_data.items():
                            # Format the GO term entry
                            entry = self.format_go_entry(
                                sequence_id, 
                                data['sequence'], 
                                data['go_annotations']
                            )
                            
                            # Save position for this entry
                            positions.append(current_position)
                            
                            # Map UniRef ID to index
                            id_to_index[sequence_id] = sequence_index
                            sequence_index += 1
                            
                            # Write to file and track position
                            outfile.write(entry)
                            outfile.write('\n\n')  # Add extra newline to separate entries
                            
                            # Update position for next entry
                            current_position += len(entry) + 2  # +2 for the extra newlines
                            
                            # Flush to ensure positions are accurate
                            outfile.flush()
                    
                    except Exception as e:
                        print(f"Error processing {blob.name}: {e}")
                        continue
        
        # Add an extra position marker at the end (to make slicing work)
        positions.append(current_position)
        
        # Save the positions as a NumPy array
        np.save(positions_path, np.array(positions, dtype=np.int64))
        
        # Save the UniRef ID to index mapping as a text file
        with open(id_to_index_path, 'w') as f:
            # Each line contains: UniRef_ID\tIndex
            for uniref_id, idx in id_to_index.items():
                f.write(f"{uniref_id}\t{idx}\n")

        # Also save as JSON for easier inspection/debugging
        with open(f"{id_to_index_path}.json", 'w') as f:
            json.dump(id_to_index, f)

        print(f"\nSuccessfully processed {sequence_index} sequences with GO terms")
        print(f"GO terms data saved to: {go_terms_path}")
        print(f"Position index saved to: {positions_path}")
        print(f"UniRef ID to index mapping saved to: {id_to_index_path}")

        return sequence_index
    
    def read_go_terms_by_index(self, idx, go_terms_file="go_terms.txt", positions_file="go_positions.idx.npy"):
        """
        Read GO terms data for a sequence by its index.
        
        Args:
            idx: Sequence index
            go_terms_file: Name of the GO terms text file
            positions_file: Name of the positions index file
            
        Returns:
            GO terms data for the specified sequence
        """
        go_terms_path = os.path.join(self.output_dir, go_terms_file)
        positions_path = os.path.join(self.index_dir, positions_file)
        
        # Load the position index
        positions = np.load(positions_path)
        
        # Check if index is valid
        if idx < 0 or idx >= len(positions) - 1:
            raise ValueError(f"Index out of range. Valid range: 0-{len(positions) - 2}")
        
        # Get the start and end positions
        start = positions[idx]
        end = positions[idx + 1]
        
        # Read the data from the text file
        with open(go_terms_path, 'r') as f:
            f.seek(start)
            data = f.read(end - start)
        
        return data
    
    def read_go_terms_by_uniref_id(self, uniref_id, go_terms_file="go_terms.txt", 
                                 positions_file="go_positions.idx.npy", 
                                 id_to_index_file="uniref_to_index.txt"):
        """
        Read GO terms data for a sequence by its UniRef ID.
        
        Args:
            uniref_id: UniRef ID of the sequence
            go_terms_file: Name of the GO terms text file
            positions_file: Name of the positions index file
            id_to_index_file: Name of the UniRef ID to index mapping file
            
        Returns:
            GO terms data for the specified sequence or None if not found
        """
        id_to_index_path = os.path.join(self.index_dir, id_to_index_file)
        
        # Try the JSON version first (for development/debugging)
        try:
            with open(f"{id_to_index_path}.json", 'r') as f:
                id_to_index = json.load(f)
            
            # Convert keys to proper format if needed
            if isinstance(next(iter(id_to_index.keys()), None), str):
                index = id_to_index.get(uniref_id)
            else:
                index = id_to_index.get(str(uniref_id))
            
            if index is not None:
                return self.read_go_terms_by_index(index, go_terms_file, positions_file)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        # Read from the text mapping file
        try:
            with open(id_to_index_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split('\t')
                    if len(parts) == 2:
                        current_id, idx_str = parts
                        
                        if current_id == uniref_id:
                            try:
                                idx = int(idx_str)
                                return self.read_go_terms_by_index(idx, go_terms_file, positions_file)
                            except ValueError:
                                print(f"Invalid index value for ID {current_id}: {idx_str}")
        except FileNotFoundError:
            print(f"ID to index mapping file not found: {id_to_index_path}")
        
        # ID not found
        return None


class GOTermsDataset:
    """
    A dataset class for accessing GO terms data using the text file and indexes.
    """
    
    def __init__(self, go_terms_path, positions_path):
        """
        Initialize the dataset.
        
        Args:
            go_terms_path: Path to the GO terms text file
            positions_path: Path to the positions index file
        """
        self.go_terms_path = go_terms_path
        self.positions_path = positions_path
        self.need_setup = True
        self.lock = threading.Lock()
    
    def setup(self):
        """Initial lightweight setup"""
        self.file = None
        self.positions = np.load(self.positions_path, mmap_mode='r')  # Memory-mapped for efficiency
        self.need_setup = True
    
    def actual_setup(self):
        """Actual setup that opens the file"""
        self.file = open(self.go_terms_path, 'r')
        if not hasattr(self, 'positions'):
            self.positions = np.load(self.positions_path, mmap_mode='r')
        self.need_setup = False
    
    def __len__(self):
        """Get the number of sequences in the dataset"""
        if not hasattr(self, 'positions'):
            self.positions = np.load(self.positions_path, mmap_mode='r')
        
        return len(self.positions) - 1  # Skip last position marker
    
    def __getitem__(self, idx):
        """Get GO terms data for a sequence by its index"""
        if self.need_setup:
            self.actual_setup()
        
        # Check if index is valid
        if idx < 0 or idx >= len(self.positions) - 1:
            raise IndexError(f"Index out of range: {idx}")
        
        # Get the start and end positions
        start = self.positions[idx]
        end = self.positions[idx + 1]
        
        # Read the data from the text file
        with self.lock:
            self.file.seek(start)
            data = self.file.read(end - start)
        
        # Parse the data
        lines = data.strip().split('\n')
        
        # Parse the formatted data
        sequence_id = lines[0][1:]  # Remove '>' prefix
        sequence = lines[1]
        
        # Parse GO annotations
        go_annotations = []
        for i in range(2, len(lines)):
            if not lines[i].strip():  # Skip empty lines
                continue
                
            parts = lines[i].split()
            if len(parts) >= 3:
                go_id = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                go_annotations.append({
                    'go_id': go_id,
                    'start': start,
                    'end': end
                })
        
        # Return a dictionary with the parsed data
        return {
            'sequence_id': sequence_id,
            'sequence': sequence,
            'go_annotations': go_annotations
        }
    
    def close(self):
        """Close the file if it's open"""
        if hasattr(self, 'file') and self.file:
            self.file.close()
            self.file = None


# Utility function to search for UniRef IDs in the mapping file
def find_uniref_id_index(uniref_id, id_to_index_path):
    """
    Find the index of a UniRef ID in the mapping file.
    
    Args:
        uniref_id: UniRef ID to search for
        id_to_index_path: Path to the ID to index mapping file
        
    Returns:
        Index of the UniRef ID or None if not found
    """
    try:
        # Try the JSON version first (faster for in-memory lookup)
        with open(f"{id_to_index_path}.json", 'r') as f:
            id_to_index = json.load(f)
            return id_to_index.get(uniref_id)
    except:
        pass
    
    # Fall back to text file search
    try:
        with open(id_to_index_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2 and parts[0] == uniref_id:
                    return int(parts[1])
    except:
        pass
    
    return None


# Example usage
def main():
    # Configure extractor
    BUCKET_NAME = "interproscan"
    BASE_PREFIX = "output3/folder_"
    
    # For testing, you might want to limit the range
    START_FOLDER = 0
    END_FOLDER = 704  # Set to 704 for full processing
    
    # Initialize the extractor
    extractor = GOTermsExtractor(
        bucket_name=BUCKET_NAME,
        base_prefix=BASE_PREFIX,
        start_folder=START_FOLDER,
        end_folder=END_FOLDER
    )
    
    # Process all folders
    extractor.process_all_folders()
    
    # Example: Read GO terms for a sequence by index
    go_terms_data = extractor.read_go_terms_by_index(10000000)
    print("\nSample GO terms data (by index):")
    print(go_terms_data[:500] + "..." if len(go_terms_data) > 500 else go_terms_data)
    
    # Example: Create and use the dataset
    dataset = GOTermsDataset(
        go_terms_path=os.path.join(extractor.output_dir, "go_terms.txt"),
        positions_path=os.path.join(extractor.index_dir, "go_positions.idx.npy")
    )
    
    # Get a sample entry
    entry = dataset[0]
    print("\nSample entry from dataset:")
    print(f"Sequence ID: {entry['sequence_id']}")
    print(f"Sequence: {entry['sequence'][:50]}..." if len(entry['sequence']) > 50 else entry['sequence'])
    print(f"GO annotations: {entry['go_annotations'][:3]}..." if len(entry['go_annotations']) > 3 else entry['go_annotations'])
    
    # Example: Look up a UniRef ID
    id_to_index_path = os.path.join(extractor.index_dir, "uniref_to_index.txt")
    sample_id = list(entry.values())[0]  # Get first UniRef ID for demo
    idx = find_uniref_id_index(sample_id, id_to_index_path)
    if idx is not None:
        print(f"\nFound UniRef ID {sample_id} at index {idx}")
    
    # Clean up
    dataset.close()


if __name__ == "__main__":
    main()