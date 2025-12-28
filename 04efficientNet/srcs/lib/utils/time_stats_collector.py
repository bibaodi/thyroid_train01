import time


class TimeStatsCollector:
    def __init__(self):
        # Initialize time statistics variables
        self.total_start_time = 0
        self.total_end_time = 0
        self.total_prediction_time = 0
        self.total_processing_time = 0
        self.case_times = []  # Store time information for each case
        self.image_count = 0  # Total number of images processed
        self.case_count = 0   # Total number of cases processed
        
    def start_total_timer(self):
        """Start total running time timer"""
        self.total_start_time = time.perf_counter()
    
    def end_total_timer(self):
        """End total running time timer"""
        self.total_end_time = time.perf_counter()
    
    def start_prediction_timer(self):
        """Start model prediction time timer"""
        return time.perf_counter()
    
    def end_prediction_timer(self, start_time):
        """End model prediction time timer and accumulate"""
        self.total_prediction_time += time.perf_counter() - start_time
    
    def start_processing_timer(self):
        """Start result processing time timer"""
        return time.perf_counter()
    
    def end_processing_timer(self, start_time):
        """End result processing time timer and accumulate"""
        self.total_processing_time += time.perf_counter() - start_time
    
    def record_case_time(self, case_name, case_time, image_count):
        """Record processing time for a single case"""
        self.case_times.append({
            'case_name': case_name,
            'time_taken': case_time,
            'image_count': image_count
        })
        self.case_count += 1
        self.image_count += image_count
    
    def get_total_time(self):
        """Get total running time"""
        return self.total_end_time - self.total_start_time
    
    def calculate_statistics(self):
        """Calculate statistics"""
        if self.case_count == 0:
            return {}
        
        # Calculate total time
        total_time = self.get_total_time()
        # Calculate average time per case
        avg_case_time = total_time / self.case_count if self.case_count > 0 else 0
        # Calculate average time per image
        avg_image_time = total_time / self.image_count if self.image_count > 0 else 0
        # Calculate average prediction time ratio
        pred_time_ratio = (self.total_prediction_time / total_time) * 100 if total_time > 0 else 0
        # Calculate average processing time ratio
        proc_time_ratio = (self.total_processing_time / total_time) * 100 if total_time > 0 else 0
        
        return {
            'total_time': total_time,
            'avg_case_time': avg_case_time,
            'avg_image_time': avg_image_time,
            'prediction_time': self.total_prediction_time,
            'processing_time': self.total_processing_time,
            'pred_time_ratio': pred_time_ratio,
            'proc_time_ratio': proc_time_ratio,
            'case_count': self.case_count,
            'image_count': self.image_count
        }
    
    def print_statistics(self):
        """Print statistics"""
        stats = self.calculate_statistics()
        
        if not stats:
            print("No statistics available.")
            return
        
        print("\n=========== Time Statistics ===========")
        print(f"Total running time: {stats['total_time']:.4f} seconds")
        print(f"Total cases processed: {stats['case_count']}")
        print(f"Total images processed: {stats['image_count']}")
        print(f"Average time per case: {stats['avg_case_time']:.4f} seconds")
        print(f"Average time per image: {stats['avg_image_time']:.4f} seconds")
        print(f"Total prediction time: {stats['prediction_time']:.4f} seconds ({stats['pred_time_ratio']:.2f}%)")
        print(f"Total processing time: {stats['processing_time']:.4f} seconds ({stats['proc_time_ratio']:.2f}%)")
        print("======================================")
        
        # Print detailed time information for each case
        if 0 and self.case_times:
            print("\nCase-wise Time Details:")
            print("--------------------------------------")
            print("{:<20} {:>12} {:>12} {:>12}".format("Case Name", "Time (s)", "Images", "Avg/Image (s)"))
            print("--------------------------------------")
            
            for case in self.case_times:
                avg_per_image = case['time_taken'] / case['image_count'] if case['image_count'] > 0 else 0
                print("{:<20} {:>12.4f} {:>12} {:>12.4f}".format(
                    case['case_name'][:19],  # Limit length to maintain format
                    case['time_taken'],
                    case['image_count'],
                    avg_per_image
                ))
            print("--------------------------------------")