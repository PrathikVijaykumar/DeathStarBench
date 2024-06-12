import os
import sys
import grpc
import pickle
import requests
import datetime
from google.protobuf.timestamp_pb2 import Timestamp
import statistics

sys.path.insert(0, os.getcwd() + '/proto_gen_python')
from proto_gen_python import query_pb2, query_pb2_grpc


def json_query_service():
    URL = 'http://localhost:16686/api'
    values = requests.get(URL + '/services').json()['data']
    print(values)


def create_grpc_channel():
    channel = grpc.insecure_channel('localhost:16685')
    return channel


def create_grpc_stub(channel):
    stub = query_pb2_grpc.QueryServiceStub(channel)
    return stub


def grpc_query_service(stub):
    response = stub.GetServices(query_pb2.GetServicesRequest())
    print(response)


# Construct trace_dict from a list of spans
# trace: List of different spans
# trace_id: ID of the trace
# Returns a dictionary with the following format:
# {
#   'trace_id': <trace_id>,
#   'spans': {
#       <service_name>: <duration>
#   }
# }
def construct_trace(trace, trace_id):
    trace_dict = {'trace_id': trace_id, 'spans': {}}

    # NOTE: Currently not subtracting latencies
    # # Sort trace by durations
    # trace.sort(key=lambda x: x[2])

    # # Subtract child spans from parent spans
    # for i in range(len(trace)):
    #     span_details = trace[i]
    #     parent_span = span_details[3]
    #     search_limit = i

    #     # Find parent service and subtract duration
    #     flag = False
    #     while parent_span != None:
    #         for j in range(search_limit, len(trace)):
    #             if trace[j][0] == parent_span:
    #                 trace[j][2] -= span_details[2]
    #                 parent_span = trace[j][3]
    #                 search_limit = j
    #                 break

    # Construct trace dict
    for span_details in trace:
        service = span_details[1]
        if service not in trace_dict['spans']:
            trace_dict['spans'][service] = span_details[2]
        else:
            trace_dict['spans'][service] = max(trace_dict['spans'][service],
                                               span_details[2])

    return trace_dict


# Query traces of a service
# service_name: name of the service
# time: time in minutes
# log: whether to log the traces
# req_name: name of the trace
def grpc_query_traces(stub,
                      service_name='compose-post-service',
                      log=True,
                      req_name='home'):
    # Read the start time from the file
    # home = os.environ['HOME']
    # with open(os.path.join(home, 'start_time.pkl'), 'rb') as f:
    #     start_time = pickle.load(f)
    now = datetime.datetime.now()
    start_time = Timestamp(seconds=int(now.timestamp())-30) #300

    # Construct the TraceQueryParameters object
    trace_query_parameters = query_pb2.TraceQueryParameters(
        service_name=service_name,
        start_time_min=start_time,
        search_depth=50000,
    )

    # Construct the FindTracesRequest object
    find_traces_request = query_pb2.FindTracesRequest(
        query=trace_query_parameters)

    stream_responses = stub.FindTraces(find_traces_request)

    # Append all traces in a list
    all_traces = []

    # Construct traces from the stream responses
    curr_trace_id = None
    trace = []  # List of span details
    svc_count = {}  # Count how many times a service was present in the trace

    #-----------
    # total_duration = 0  # To store the sum of durations
    count = 0  # To count the number of compose-post-service spans
    durations = []  # List to store durations
    #-----------

    service_durations = {}
    service_counts = {}

    stream_responses = stub.FindTraces(find_traces_request)

    for response in stream_responses:
        for span in response.spans:  ### Use the span with max duration in the trace
            service = span.process.service_name
            duration = span.duration.nanos / 1000  # Convert duration from nanoseconds to microseconds

            if service not in service_durations:
                service_durations[service] = duration
                service_counts[service] = 1
            else:
                service_durations[service] += duration
                service_counts[service] += 1

    # Calculate average for each service and sum them up
    total_average_duration = 0
    for service, total_duration in service_durations.items():
        average_duration = total_duration / service_counts[service]
        print(f'Average duration for {service}: {average_duration} microseconds')
        total_average_duration += average_duration

    print(f'Total average duration across all services: {total_average_duration} microseconds')
    # Additional statistics
        # print(f'Median duration: {statistics.median(durations)} microseconds')
        # if len(durations) > 1:  # Variance and standard deviation require at least two data points
        #     print(f'Standard deviation: {statistics.stdev(durations)} microseconds')
        #     print(f'Variance: {statistics.variance(durations)}')

        # # Identify outliers - simple method based on standard deviation
        # threshold = statistics.stdev(durations) * 2
        # outliers = [d for d in durations if abs(d - average_duration) > threshold]
        # print(f'Number of outliers: {len(outliers)}')

        # Optionally, you could print all durations or outliers here
        # print("All durations:", durations)
        #print("Outliers:", outliers)
    # else:
    #     print(f'No traces found for {service_name}')

    # # ... [rest of your existing code] ...
    #     average_duration = total_duration / count
    #     print(f'Average duration for {service_name}: {average_duration} microseconds')
    # else:
    #     print(f'No traces found for {service_name}')
        
    if curr_trace_id != None:
        trace_dict = construct_trace(trace, curr_trace_id)
        for svc, _ in trace_dict['spans'].items():
            if svc not in svc_count:
                svc_count[svc] = 1
            else:
                svc_count[svc] += 1
        all_traces.append(trace_dict)

    # print('Number of request traces found for {0}: {1}'.format(
    #     service_name, len(all_traces)))
    #print('Service-wise distribution: ', svc_count)
    #print("now going to logging")
    log = True

    if log:
        #print("logging enabled")
        # Get the $HOME environment variable
        home = os.environ['HOME']
        out_dir = "/home/grads/p/prathikvijaykumar/" #os.path.join(home, 'out')

        # Save the traces to a pickle file
        with open(os.path.join(out_dir, 'traces_{0}.pkl'.format(req_name)),
                  'wb') as f:
            pickle.dump(all_traces, f)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(
            'Usage: python query_jaeger.py <grpc/json> <whether to log traces> <trace_name> '
        )
        sys.exit(1)
    if sys.argv[1] == 'grpc':
        channel = create_grpc_channel()
        stub = create_grpc_stub(channel)
        grpc_query_service(stub)
        grpc_query_traces(stub, 'compose-post-service', sys.argv[2] == 'True',
                          sys.argv[3])
    elif sys.argv[1] == 'json':
        json_query_service()