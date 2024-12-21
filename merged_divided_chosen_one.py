import os
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

import numpy as np
from math import radians, sin, cos, sqrt, atan2
import elkai  # Import elkai for TSP solving
from sklearn.cluster import KMeans  # For KMeans++ clustering
import heapq

# New function to generate clusters using KMeans with depot
def kmeans_with_depot(coordinates, num_clusters):
    depot = list(coordinates[0])
    points = [list(point) for point in coordinates[1:]]

    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    kmeans.fit(points)

    clustered_points = [[] for _ in range(num_clusters)]
    for point, label in zip(points, kmeans.labels_):
        clustered_points[label].append(point)

    result = [[depot] + [point for point in cluster] for cluster in clustered_points]

    return result

# The depot and customer coordinates
coordinates = [[41.019734, 28.81971], [41.02602, 28.82905], [41.03434, 28.8332], [41.02782, 28.8291], [41.03158, 28.82734], [41.0302, 28.82899], [41.03576, 28.82673], [41.03102, 28.82927], [41.03634, 28.83274], [41.03608, 28.83282], [41.0313, 28.82828], [41.03487, 28.82285], [41.0396, 28.82519], [41.02895, 28.82572], [41.01078, 28.82474], [41.00199, 28.83423], [40.99878, 28.83053], [41.00859, 28.82193], [41.00227, 28.83577], [41.00714, 28.81988], [41.00326, 28.83072], [41.00881, 28.83086], [41.00262, 28.8305], [41.00576, 28.82782], [40.99799, 28.83085], [41.00297, 28.83599], [41.00817, 28.8329], [40.99968, 28.83045], [41.00571, 28.83097], [41.00317, 28.83063], [41.00255, 28.83048], [41.00213, 28.83393], [41.0077, 28.81984], [41.00279, 28.83604], [40.9979, 28.83468], [41.01119, 28.82867], [41.00815, 28.85395], [41.00512, 28.8441], [41.00987, 28.85501], [41.00577, 28.8445], [41.00567, 28.84485], [41.00678, 28.85301], [40.99758, 28.84846], [41.00853, 28.84895], [40.99942, 28.84662], [41.00461, 28.84346], [41.00515, 28.84431], [40.99917, 28.85226], [41.00649, 28.84896], [41.0108, 28.85005], [41.0405, 28.84485], [41.03921, 28.84449], [41.02525, 28.83666], [41.03086, 28.84659], [41.0306, 28.83904], [41.03525, 28.84121], [41.04448, 28.83297], [41.03153, 28.84416], [41.03143, 28.844], [41.04364, 28.84111], [41.03075, 28.83783], [41.0549, 28.82542], [41.06006, 28.8358], [41.046, 28.85154], [41.05421, 28.84883], [41.05672, 28.84225], [41.04435, 28.82023], [41.0544, 28.84569], [41.04434, 28.81931], [41.10403, 28.86451], [41.04713, 28.85006], [41.05089, 28.84942], [41.04835, 28.85265], [41.05632, 28.84585], [41.03983, 28.81678], [41.06019, 28.83569], [41.00878, 28.87152], [41.00664, 28.86902], [41.01079, 28.87049], [41.00642, 28.87068], [41.0155, 28.87553], [41.01081, 28.86759], [41.00577, 28.86982], [40.99978, 28.8703], [41.00144, 28.874], [41.00336, 28.87093], [41.00447, 28.88763], [41.00609, 28.87018], [41.00649, 28.89176], [41.00636, 28.88403], [41.00687, 28.86951], [41.09623, 28.7909], [41.08429, 28.77111], [41.10239, 28.78701], [41.09484, 28.77081], [41.10226, 28.78703], [41.0951, 28.77448], [41.10127, 28.78728], [41.02142, 28.83977], [41.11218, 28.78581], [41.10364, 28.79214], [41.09398, 28.775], [41.06501, 28.79226], [41.11199, 28.78498], [41.03396, 28.79644], [41.03354, 28.77447], [41.0357, 28.78919], [41.03783, 28.7983], [41.03664, 28.78836], [41.03355, 28.80022], [41.0327, 28.80025], [41.03337, 28.80022], [41.03386, 28.80313], [41.03783, 28.79847], [41.03703, 28.78402], [41.03311, 28.79984], [41.04362, 28.78395], [41.03544, 28.79396], [41.05943, 28.79879], [41.0511, 28.79667], [41.05471, 28.79464], [41.047, 28.8004], [41.05481, 28.79922], [41.05732, 28.80775], [41.05702, 28.80761], [41.05296, 28.80591], [41.05771, 28.80125], [41.05284, 28.79818], [41.06106, 28.80147], [41.05203, 28.79875], [41.05811, 28.79877], [41.04828, 28.8008], [41.05029, 28.80319], [41.05404, 28.79501], [41.11479, 28.81034], [41.11022, 28.80399], [41.10704, 28.78471], [41.09056, 28.81102], [41.10078, 28.81058], [41.11002, 28.80115], [41.09743, 28.80579], [41.10905, 28.78911], [41.10651, 28.78756], [41.10802, 28.78975], [41.10679, 28.80359], [41.12037, 28.80708], [41.12019, 28.8075], [41.10855, 28.80333], [40.97926, 28.85509], [40.9774, 28.87706], [40.97926, 28.8551], [40.97849, 28.87509], [40.97914, 28.87355], [40.97869, 28.87964], [40.97806, 28.87267], [40.97829, 28.87212], [40.97769, 28.8769], [40.97892, 28.87213], [40.97966, 28.87484], [40.97819, 28.87171], [40.98343, 28.86834], [40.99183, 28.83519], [40.98926, 28.87012], [40.98249, 28.87297], [40.98732, 28.86599], [40.98754, 28.8661], [40.98105, 28.87006], [40.9868, 28.86887], [41.11845, 28.77374], [41.11283, 28.76782], [41.11711, 28.77373], [41.13167, 28.78009], [41.11739, 28.77048], [41.12238, 28.7703], [41.11923, 28.76654], [41.11711, 28.77347], [41.12251, 28.77141], [41.1262, 28.77207], [41.12395, 28.77161], [41.1227, 28.77899], [41.12066, 28.7683], [41.02245, 28.83912], [41.01965, 28.82418], [41.02255, 28.8392], [41.02748, 28.85722], [41.01776, 28.84158], [41.01831, 28.83251], [41.02941, 28.85426], [41.02428, 28.85081], [41.01485, 28.84199], [41.03148, 28.85003], [41.0517, 28.84262], [41.01786, 28.82593], [41.01465, 28.83501], [41.01433, 28.83542], [41.0169, 28.84888], [41.04775, 28.84593], [41.01456, 28.83531], [41.04588, 28.84645], [41.04919, 28.83479], [41.01604, 28.82458], [41.04896, 28.83851], [40.98085, 28.79397], [40.96996, 28.79621], [40.96421, 28.83819], [40.95922, 28.83478], [40.97259, 28.80423], [40.98526, 28.79617], [40.95851, 28.82128], [40.96392, 28.83777], [40.96379, 28.83763], [40.97981, 28.79449], [41.00128, 28.77555], [40.99954, 28.78], [40.99861, 28.78637], [40.99498, 28.7912], [40.99393, 28.79101], [40.99274, 28.78692], [40.99824, 28.76625], [41.00056, 28.79316], [40.99404, 28.76789], [40.99699, 28.77676], [41.00121, 28.79735], [40.99928, 28.78542], [40.99659, 28.77578], [40.99812, 28.77808], [40.99623, 28.79155], [41.04447, 28.76268], [41.05557, 28.78605], [41.05492, 28.79224], [41.04043, 28.7706], [41.05556, 28.7642], [41.05662, 28.79019], [41.0578, 28.77666], [41.05292, 28.75341], [41.05458, 28.7911], [41.05471, 28.76214], [40.98635, 28.61674], [41.00192, 28.78634], [41.00251, 28.78824], [41.01079, 28.79847], [41.0061, 28.79811], [41.0242, 28.80073], [41.01474, 28.79731], [41.02653, 28.79636], [41.00336, 28.79283], [41.02465, 28.79643], [41.0091, 28.79973], [41.00267, 28.78858], [41.00673, 28.79858], [41.00241, 28.79292], [41.01, 28.77869], [41.02842, 28.7753], [41.01749, 28.78498], [41.02205, 28.7895], [41.01366, 28.78738], [41.02528, 28.81553], [41.02004, 28.78652], [41.01749, 28.78521], [41.02173, 28.78627], [41.01147, 28.77735], [41.02058, 28.79162], [41.04509, 28.8221], [41.05049, 28.85791], [41.04183, 28.86832], [41.04149, 28.87133], [41.05047, 28.85793], [41.0447, 28.86895], [41.04187, 28.871], [41.03977, 28.87029], [41.03808, 28.85012], [41.04339, 28.86287], [41.04322, 28.85001], [41.03858, 28.87124], [41.10249, 28.76225], [41.08194, 28.75223], [41.07455, 28.74939], [41.08177, 28.75215], [41.08325, 28.75242], [41.08665, 28.75115], [41.0777, 28.75076], [41.07434, 28.75195], [41.10081, 28.76186], [41.10135, 28.76213], [41.08682, 28.75241], [40.99237, 28.88436], [40.99018, 28.87662], [40.99831, 28.86532], [40.99791, 28.86503], [40.99727, 28.86732], [40.98886, 28.87139], [40.9963, 28.86331], [40.99205, 28.88238], [40.99069, 28.88239], [40.99365, 28.87432], [40.9957, 28.86425], [40.99054, 28.87073], [40.99288, 28.88349], [41.02978, 28.86965], [41.03493, 28.8549], [41.03605, 28.86014], [41.03007, 28.86985], [41.03481, 28.8553], [41.02845, 28.86527], [41.03296, 28.86349], [41.03083, 28.86966], [41.03947, 28.85785], [41.03351, 28.86174], [41.03467, 28.86679], [41.02575, 28.86134], [41.03421, 28.85995], [40.99913, 28.8387], [41.00317, 28.84295], [40.99977, 28.84421], [41.01183, 28.83889], [41.00821, 28.83881], [40.99268, 28.84227], [41.0084, 28.84147], [40.99227, 28.84081], [40.99451, 28.84576], [41.00304, 28.84291], [40.99462, 28.8413], [41.01148, 28.83453], [40.98804, 28.78032], [40.98884, 28.78093], [40.99639, 28.77529], [40.99363, 28.77587], [40.99302, 28.77779], [40.98986, 28.7864], [40.98833, 28.78191], [40.99277, 28.78173], [40.99501, 28.77398], [40.99995, 28.79734], [40.99197, 28.77848], [40.98703, 28.78417], [41.00214, 28.77796], [40.98894, 28.7905], [40.99093, 28.78424], [40.99939, 28.79783], [41.03067, 28.87929], [41.03563, 28.88602], [41.03773, 28.87727], [41.03343, 28.88799], [41.03381, 28.88571], [41.03773, 28.87727], [41.0297, 28.89694], [41.03522, 28.88561], [41.03253, 28.87887], [41.02693, 28.88161], [41.03547, 28.87944], [41.0319, 28.87701], [41.01419, 28.85722], [41.00726, 28.86092], [41.00844, 28.85911], [41.00024, 28.8607], [41.00108, 28.85698], [41.00946, 28.85897], [41.01281, 28.85876], [41.00141, 28.85514], [41.00201, 28.86023], [41.0052, 28.85914], [41.00231, 28.8553], [41.01127, 28.86132], [40.99837, 28.85601], [40.99957, 28.85926], [41.00348, 28.85935], [40.99959, 28.85742], [41.01603, 28.8577], [41.02293, 28.85738], [41.0218, 28.85616], [41.02289, 28.85679], [41.01419, 28.86521], [41.01426, 28.8696], [41.01426, 28.86499], [41.01662, 28.86256], [41.02024, 28.85869], [41.01724, 28.86497], [41.01427, 28.86499], [41.01671, 28.86888], [41.01715, 28.86044], [41.02201, 28.85613], [41.04424, 28.87849], [41.05024, 28.8604], [41.04128, 28.88248], [41.05332, 28.85363], [41.04649, 28.87338], [41.04569, 28.88007], [41.0392, 28.87739], [41.0461, 28.87777], [41.04389, 28.8748], [41.04211, 28.87758], [41.04037, 28.88336], [41.04513, 28.82206], [41.04395, 28.87964], [41.02297, 28.87718], [41.02145, 28.87067], [41.02417, 28.87289], [41.01944, 28.89975], [41.02034, 28.87828], [41.02107, 28.87962], [41.02015, 28.87892], [41.01452, 28.88337], [41.01604, 28.89648], [41.02155, 28.87041], [41.01953, 28.87462], [41.01578, 28.8777], [40.9936, 28.85213], [40.98155, 28.85016], [40.99788, 28.85626], [40.98885, 28.84529], [40.98465, 28.84658], [40.9941, 28.85229], [40.98582, 28.84913], [40.99704, 28.85532], [40.99706, 28.85518], [40.99581, 28.85402], [40.99382, 28.85224], [41.00982, 28.81493], [40.9853, 28.83361], [40.99881, 28.88644], [41.03779, 28.83134], [41.06612, 28.82475], [41.03997, 28.81039], [41.04759, 28.80988]]  # Replace with your actual coordinates  # Replace with your actual coordinates


# Parameters
speed_km_per_hr = 35     # Speed of the vehicle (km/h)
service_time_hr = 0.05   # Service time per node (hours)
tmax = 3                 # Maximum allowed shift time in a cluster(11am-2pm)(hours)
hiring_cost_per_cluster = 50  # Hiring cost per cluster/vehicle
distance_cost_per_km = 2      # Cost per kilometer traveled

# Haversine formula to calculate distance between two points
def haversine(point1, point2):
    R = 6371  # Radius of the Earth in kilometers
    lat1, lon1 = point1
    lat2, lon2 = point2

    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Function to calculate total distance for a cluster
def calculate_total_distance(cluster, tour):
    total_distance = 0
    for i in range(len(tour)):
        point1 = cluster[tour[i]]
        point2 = cluster[tour[(i + 1) % len(tour)]]
        total_distance += haversine(point1, point2)
    return total_distance

# Function to calculate total travel time for a cluster
def calculate_total_time(cluster, tour, speed_km_per_hr, service_time_hr):
    total_distance = calculate_total_distance(cluster, tour)
    travel_time = total_distance / speed_km_per_hr
    total_service_time = (len(cluster)-1) * service_time_hr
    return travel_time + total_service_time

# Function to solve TSP using elkai with constraints
def solve_tsp_elkai_constrained(cluster, tmax, speed_km_per_hr, service_time_hr):
    # Check if the cluster has at least two points
    if len(cluster) < 2:
        print("Cluster has fewer than 2 points. Skipping TSP solving.")
        return None, None, None

    # Handle clusters with 2 points
    if len(cluster) == 2:
        # Directly calculate the distance between the two points
        total_distance = haversine(cluster[0], cluster[1])
        total_time = total_distance / speed_km_per_hr + service_time_hr
        tour = [0, 1]  # Tour is simply the two points
        return tour, total_distance, total_time

    # Create a distance matrix for the cluster
    num_points = len(cluster)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i][j] = haversine(cluster[i], cluster[j])

    # Convert the distance matrix to integers (multiply by 1000 to convert km to meters)
    distance_matrix_int = np.round(distance_matrix * 1000).astype(int)

    # Solve the TSP using elkai
    tour = elkai.solve_int_matrix(distance_matrix_int)

    # Calculate total distance and total time
    total_distance = calculate_total_distance(cluster, tour)
    total_time = calculate_total_time(cluster, tour, speed_km_per_hr, service_time_hr)

    # Check if the solution meets the time constraint
    if total_time <= tmax:
        return tour, total_distance, total_time
    else:
        print(f"Cluster exceeds time constraint: Time = {total_time:.2f} hours")
        return None, total_distance, total_time

# Function to divide a cluster into two using KMeans++
def divide_cluster_kmeans(cluster):
    # The first point is the depot
    depot = cluster[0]

    # Exclude the depot for clustering
    cluster_np = np.array(cluster[1:])

    # Apply KMeans++ with 2 clusters
    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
    kmeans.fit(cluster_np)

    # Get the labels for each point
    labels = kmeans.labels_

    # Split the cluster into two subclusters, ensuring the depot is added to both
    cluster1 = [depot] + [cluster[i + 1] for i in range(len(cluster) - 1) if labels[i] == 0]
    cluster2 = [depot] + [cluster[i + 1] for i in range(len(cluster) - 1) if labels[i] == 1]

    return cluster1, cluster2

# Function to calculate the centroid of a cluster
def calculate_centroid(cluster):
    lats = [point[0] for point in cluster]
    lons = [point[1] for point in cluster]
    centroid_lat = sum(lats) / len(cluster)
    centroid_lon = sum(lons) / len(cluster)
    return (centroid_lat, centroid_lon)

# Function to calculate the objective function value
def calculate_objective_function(total_distance, use_cluster):
    distance_cost = total_distance * distance_cost_per_km
    hiring_cost = hiring_cost_per_cluster if use_cluster else 0
    return distance_cost + hiring_cost

# Recursive function to process a cluster or sub-cluster
def process_cluster(cluster, tmax, speed_km_per_hr, service_time_hr, cluster_index=None, sub_cluster_index=None, depth=0, cluster_list=None):
    indent = "  " * depth  # Indentation for better readability
    print(f"{indent}Processing {'Sub-' if sub_cluster_index is not None else ''}Cluster {sub_cluster_index + 1 if sub_cluster_index is not None else cluster_index + 1}...")

    # Solve TSP for the current cluster with constraints
    tour, total_distance, total_time = solve_tsp_elkai_constrained(
        cluster, tmax, speed_km_per_hr, service_time_hr
    )

    # If the solution is valid, return the results
    if tour is not None:
        print(f"{indent}{'Sub-' if sub_cluster_index is not None else ''}Cluster {sub_cluster_index + 1 if sub_cluster_index is not None else cluster_index + 1} - Total Distance: {total_distance:.2f} km")
        print(f"{indent}{'Sub-' if sub_cluster_index is not None else ''}Cluster {sub_cluster_index + 1 if sub_cluster_index is not None else cluster_index + 1} - Total Time: {total_time:.2f} hours")

        # Calculate the objective function value for this cluster
        use_cluster = total_time <= tmax
        objective_value = calculate_objective_function(total_distance, use_cluster)
        print(f"{indent}{'Sub-' if sub_cluster_index is not None else ''}Cluster {sub_cluster_index + 1 if sub_cluster_index is not None else cluster_index + 1} - Objective Function Value: {objective_value:.2f}")

        # Calculate the centroid of the cluster
        centroid = calculate_centroid(cluster)

        # Add the cluster to the cluster_list
        cluster_list.append({
            't': total_time,
            'centroid': centroid,
            'members': cluster
        })

        return total_distance, objective_value, use_cluster, 1  # Return 1 to count this cluster
    else:
        print(f"{indent}{'Sub-' if sub_cluster_index is not None else ''}Cluster {sub_cluster_index + 1 if sub_cluster_index is not None else cluster_index + 1} exceeds time constraint. Dividing into two clusters...")

        # Divide the cluster into two using KMeans++
        cluster1, cluster2 = divide_cluster_kmeans(cluster)

        # If division fails (e.g., cluster has fewer than 2 points), skip further division
        if cluster1 is None or cluster2 is None:
            print(f"{indent}Cluster cannot be divided further. Skipping...")
            return 0, 0, False, 0

        # Recursively process the divided clusters
        total_distance1, objective_value1, use_cluster1, count1 = process_cluster(cluster1, tmax, speed_km_per_hr, service_time_hr, cluster_index, 1, depth + 1, cluster_list)
        total_distance2, objective_value2, use_cluster2, count2 = process_cluster(cluster2, tmax, speed_km_per_hr, service_time_hr, cluster_index, 2, depth + 1, cluster_list)

        # Combine the results
        total_distance = total_distance1 + total_distance2
        objective_value = objective_value1 + objective_value2
        use_cluster = use_cluster1 or use_cluster2
        total_count = count1 + count2  # Sum the counts of sub-clusters

        return total_distance, objective_value, use_cluster, total_count

# Main function to process clusters and return the cluster_list
def main(num_clusters):
    total_system_distance = 0  # Total distance for all clusters
    total_system_cost = 0      # Total cost for all clusters
    total_hiring_cost = 0      # Total hiring cost for all clusters
    total_clusters_used = 0    # Number of clusters used
    cluster_list = []          # List to store clusters and their total times

    # Generate clusters using KMeans with depot
    clusters = kmeans_with_depot(coordinates, num_clusters)

    # Process each cluster
    for cluster_index, cluster in enumerate(clusters):
        total_distance, objective_value, use_cluster, cluster_count = process_cluster(cluster, tmax, speed_km_per_hr, service_time_hr, cluster_index, cluster_list=cluster_list)

        # Update system-wide totals
        total_system_distance += total_distance
        total_system_cost += objective_value
        if use_cluster:
            total_hiring_cost += hiring_cost_per_cluster * cluster_count
            total_clusters_used += cluster_count

    # Print the system-wide totals
    print("\nSystem-Wide Summary:")
    print(f"Total Distance for All Clusters: {total_system_distance:.2f} km")
    print(f"Total Hiring Cost for All Clusters: {total_hiring_cost:.2f}")
    print(f"Total Objective Function Value (Overall Cost): {total_system_cost:.2f}")
    print(f"Total Clusters Used: {total_clusters_used}")

    # Print and return the cluster_list
    print("\nClusters and Their Total Times:")
    for cluster in cluster_list:
        print(f"t: {cluster['t']:.2f} hours, Centroid: {cluster['centroid']}, Members: {cluster['members']}")

    # Return the cluster_list
    return cluster_list, total_system_distance, total_hiring_cost, total_system_cost, total_clusters_used

# Merging algorithm
def merge_clusters(cluster_list, tmax, speed_km_per_hr, service_time_hr):
    # Sort the clusters by their t values in ascending order
    sorted_clusters = sorted(cluster_list, key=lambda x: x['t'])

    # Function to calculate the centroid distance between two clusters
    def centroid_distance(cluster1, cluster2):
        lat1, lon1 = cluster1['centroid']
        lat2, lon2 = cluster2['centroid']
        return haversine((lat1, lon1), (lat2, lon2))

    # Function to merge two clusters
    def merge_two_clusters(cluster1, cluster2):
        merged_members = [cluster1['members'][0]] + cluster1['members'][1:] + cluster2['members'][1:]
        merged_centroid = calculate_centroid(merged_members)
        return {
            'centroid': merged_centroid,
            'members': merged_members
        }

    # Main merging loop
    merged_clusters = []
    while len(sorted_clusters) > 1:
        current_cluster = sorted_clusters.pop(0)

        # Find the 3 nearest clusters based on centroid distance
        nearest_clusters = heapq.nsmallest(3, sorted_clusters, key=lambda x: centroid_distance(current_cluster, x))

        # Try merging with the nearest clusters
        merged = False
        for nearest in nearest_clusters:
            if current_cluster['t'] + nearest['t'] <= 3:  # Check if the sum of t values is within the limit
                merged_cluster = merge_two_clusters(current_cluster, nearest)

                # Solve the TSP for the merged cluster
                tour, total_distance, total_time = solve_tsp_elkai_constrained(
                    merged_cluster['members'], tmax, speed_km_per_hr, service_time_hr
                )

                # Check if the merged cluster meets the tmax constraint
                if total_time <= tmax:
                    # Add the merged cluster to the final list
                    merged_clusters.append({
                        't': total_time,
                        'centroid': merged_cluster['centroid'],
                        'members': merged_cluster['members']
                    })
                    # Remove the nearest cluster from the sorted list
                    sorted_clusters.remove(nearest)
                    merged = True
                    break  # Stop after the first successful merge

        if not merged:
            # If no successful merge, add the current cluster to the final list
            merged_clusters.append(current_cluster)

    # Add the last remaining cluster if any
    if sorted_clusters:
        merged_clusters.append(sorted_clusters[0])

    return merged_clusters

# Call the main function and get the cluster_list
if __name__ == "__main__":
    # Define the range for num_clusters
    num_clusters_range = range(13, 18)  # 13 to 17 inclusive

    # Store results for each num_clusters
    results = []

    for num_clusters in num_clusters_range:
        print(f"\nProcessing for num_clusters = {num_clusters}")
        cluster_list, total_system_distance, total_hiring_cost, total_system_cost, total_clusters_used = main(num_clusters)
        merged_cluster_list = merge_clusters(cluster_list, tmax, speed_km_per_hr, service_time_hr)

        # Calculate and print the summary after merging
        merged_total_system_distance = 0
        merged_total_hiring_cost = 0
        merged_total_system_cost = 0
        merged_total_clusters_used = 0

        for cluster in merged_cluster_list:
            tour, total_distance, total_time = solve_tsp_elkai_constrained(
                cluster['members'], tmax, speed_km_per_hr, service_time_hr
            )
            merged_total_system_distance += total_distance
            merged_total_hiring_cost += hiring_cost_per_cluster if total_time <= tmax else 0
            merged_total_system_cost += calculate_objective_function(total_distance, total_time <= tmax)
            if total_time <= tmax:
                merged_total_clusters_used += 1

        print(f"\nSummary After Merging for num_clusters = {num_clusters}:")
        print(f"Total Distance for All Clusters: {merged_total_system_distance:.2f} km")
        print(f"Total Hiring Cost for All Clusters: {merged_total_hiring_cost:.2f}")
        print(f"Total Objective Function Value (Overall Cost): {merged_total_system_cost:.2f}")
        print(f"Total Clusters Used: {merged_total_clusters_used}")

        # Store the results
        results.append({
            'num_clusters': num_clusters,
            'total_distance': merged_total_system_distance,
            'total_hiring_cost': merged_total_hiring_cost,
            'total_system_cost': merged_total_system_cost,
            'total_clusters_used': merged_total_clusters_used,
            'merged_cluster_list': merged_cluster_list
        })

    # Print the final results for all num_clusters
    print("\nFinal Results for All num_clusters:")
    for result in results:
        print(f"num_clusters: {result['num_clusters']}, "
              f"Total Distance: {result['total_distance']:.2f} km, "
              f"Total Hiring Cost: {result['total_hiring_cost']:.2f}, "
              f"Total System Cost: {result['total_system_cost']:.2f}, "
              f"Total Clusters Used: {result['total_clusters_used']}")

    # Find the result with the lowest total system cost
    best_result = min(results, key=lambda x: x['total_system_cost'])

    # Print the best result
    print("\nBest Result (Lowest Total System Cost):")
    print(f"num_clusters: {best_result['num_clusters']}, "
          f"Total Distance: {best_result['total_distance']:.2f} km, "
          f"Total Hiring Cost: {best_result['total_hiring_cost']:.2f}, "
          f"Total System Cost: {best_result['total_system_cost']:.2f}, "
          f"Total Clusters Used: {best_result['total_clusters_used']}")

    # Print the clusters for the best result in a list format
    best_clusters = [cluster['members'] for cluster in best_result['merged_cluster_list']]
    print("\nClusters for the Best Result:")
    print(best_clusters)
