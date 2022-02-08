# Author: Conner Warnock
# This program takes a conditional probability matrix and computes mutual information, variance, channel capacity
# source/sink entropy, conditional entropy, equivocation, and system mutual information
# October 25, 2020

import scipy.io
import math
import statistics
import random
import copy

mat_file = scipy.io.loadmat('ee6743_test1.mat')
print(mat_file['a'])


# Create conditional prob matrix
def get_cond_prob(mat_file):
    cond_prob = []
    cond_prob_row = []
    for i in range(0, len(mat_file['a'])):
        cond_prob_row.clear()
        for j in range(0, len(mat_file['a'][i])):
            cond_prob_row.append(mat_file['a'][i][j])
        cond_prob.append(cond_prob_row[:])

    return cond_prob


# Checks if row probs sum to one
def check_rows_sum_to_one(cond_prob):
    rows_sum_to_one = True
    for i in range(0, len(cond_prob)):
        row_sum = 0
        for j in range(0, len(cond_prob[0])):
            row_sum = row_sum + cond_prob[i][j]
        # For some reason all these files read in what I assume is some small error (~10^-16). Don't know why this is
        if row_sum > 1.0000000001 or row_sum < 0.9999999999:
            rows_sum_to_one = False
    if rows_sum_to_one:
        print("Rows Sum to One: True")
    else:
        print("Rows Sum to One: False\nExiting.",)

    return rows_sum_to_one

# Creates a list of p(a) with them as equal. This is chosen to be the initial setting because it is a common solution
def create_initial_pa_list(cond_prob):
    pa_list = []
    for i in range(0, len(cond_prob)):
        pa_list.append(1/(len(cond_prob)))

    return pa_list


# Calculates p(bj) = sum( p(ai) * p(bj|ai) ), in other words calculates p(b) from p(a) and p(b|a)
def get_pb(cond_prob, pa_list):
    pb_list = []
    for j in range(0, len(cond_prob[0])):
        pb = 0
        for i in range(0, len(cond_prob)):
            pb = pb + pa_list[i]*cond_prob[i][j]
        pb_list.append(pb)

    return pb_list


# Calculates mutual information I(a;B) for each p(a)
# I(a;B) = sum( p(b|a)*log2( p(b|a) / p(b)  ) )
def get_mutual_info_list(cond_prob, pa_list, pb_list):
    mutual_info_list = []
    for i in range(0, len(cond_prob)):
        mutual_info = 0
        for j in range(0, len(cond_prob[i])):
            # Deals with 0 values
            if abs(cond_prob[i][j]) > 0.0 and abs(pb_list[j]) > 0.0:
                mutual_info = mutual_info + cond_prob[i][j]*math.log2(cond_prob[i][j] / pb_list[j])
        mutual_info_list.append(mutual_info)

    return mutual_info_list


# Raise/lower p(a) of highest/lowest mutual information values for 'a'
# We can increase the probability p(a) of the highest mutual information value to lower its uncertainty
# Vice versa for the lowest mutual information value
# This has the effect of "balancing" the mutual information values, while also maintaining the sum to be equal to 1
def alter_pa_list(pa_list, mutual_info_list):
    increment = 0.00001
    # Find largest/smallest mutual info index
    largest_mutual_info = -100000000
    largest_index = 0
    smallest_mutual_info = 100000000
    smallest_index = 0
    for i in range(0, len(mutual_info_list)):
        if mutual_info_list[i] > largest_mutual_info:
            largest_mutual_info = mutual_info_list[i]
            largest_index = i
        if mutual_info_list[i] < smallest_mutual_info:
            smallest_mutual_info = mutual_info_list[i]
            smallest_index = i
    pa_list[largest_index] = pa_list[largest_index] + increment
    pa_list[smallest_index] = pa_list[smallest_index] - increment

    return pa_list, increment


# Calculates channel capacity
def get_channel_capacity(mutual_info_list):
    channel_capacity = 0
    for i in range(0, len(mutual_info_list)):
        channel_capacity = channel_capacity + mutual_info_list[i]
    channel_capacity = channel_capacity /  len(mutual_info_list)

    return channel_capacity


# Calculates source entropy
def get_source_entropy(pa_list, cond_prob):
    source_entropy = 0
    conditional_entropies = []
    # Calculate conditional entropies
    for i in range(0, len(cond_prob)):
        conditional_entropy = 0
        for j in range(0, len(cond_prob[i])):
            if abs(cond_prob[i][j]) > 0:
                conditional_entropy = conditional_entropy + ( cond_prob[i][j] * (math.log2(1 / cond_prob[i][j]) ) )
        conditional_entropies.append(conditional_entropy)
    # Calculate source entropy
    for i in range(0, len(pa_list)):
        source_entropy = source_entropy + (pa_list[i] * conditional_entropies[i])

    return source_entropy


# Calculates joint pdf
def get_joint_pdf(pa_list, cond_prob):
    joint_pdf = []
    joint_pdf_row = []
    for i in range(0, len(cond_prob)):
        joint_pdf_row.clear()
        for j in range(0, len(cond_prob[i])):
            joint_probability = pa_list[i] * cond_prob[i][j]
            joint_pdf_row.append(joint_probability)
        joint_pdf.append(joint_pdf_row[:])

    return joint_pdf


# Get p(a|b) matrix
def get_inverse_cond_prob(pb_list, joint_pdf):
    inverse_cond_prob = []
    inverse_cond_prob_row = []
    for i in range(0, len(joint_pdf)):
        inverse_cond_prob_row.clear()
        for j in range(0, len(joint_pdf[i])):
            inverse_cond_prob_row.append( joint_pdf[i][j] / pb_list[j] )
        inverse_cond_prob.append(inverse_cond_prob_row[:])
    # Invert matrix
    temp = []
    temp_row = []
    for i in range(0, len(inverse_cond_prob[0])):
        temp_row.clear()
        for j in range(0, len(inverse_cond_prob)):
            temp_row.append(inverse_cond_prob[j][i])
        temp.append(temp_row[:])
    inverse_cond_prob = copy.deepcopy(temp)

    return inverse_cond_prob


# Calculate conditional entropy
def get_conditional_entropy(joint_pdf, cond_prob, reverse):
    conditional_entropy_given_A  = 0
    if len(cond_prob) == len(cond_prob[0]):
        for i in range(0, len(joint_pdf)):
            for j in range(0, len(joint_pdf[i])):
                if abs(cond_prob[i][j]) > 0:
                    conditional_entropy_given_A = conditional_entropy_given_A + ( joint_pdf[i][j] * ( math.log2(1 / cond_prob[i][j]) ) )
    else:
        if reverse == False:
            for i in range(0, len(joint_pdf)):
                for j in range(0, len(joint_pdf[i])):
                    if abs(cond_prob[i][j]) > 0:
                        conditional_entropy_given_A = conditional_entropy_given_A + ( joint_pdf[i][j] * ( math.log2(1 / cond_prob[i][j]) ) )
        else:
            for i in range(0, len(cond_prob)):
                for j in range(0, len(cond_prob[i])):
                    if abs(cond_prob[i][j]) > 0:
                        conditional_entropy_given_A = conditional_entropy_given_A + ( joint_pdf[j][i] * ( math.log2(1 / cond_prob[i][j]) ) )

    return conditional_entropy_given_A


# Calculates equivocation
def get_equivocation(joint_pdf):
    equivocation = 0
    for i in range(0, len(joint_pdf)):
        for j in range(0, len(joint_pdf[i])):
            if abs(joint_pdf[i][j]) > 0:
                equivocation = equivocation + ( joint_pdf[i][j] * ( math.log2(1 / joint_pdf[i][j]) ) )

    return equivocation


# Calculates system mutual information
def get_system_mutual_info(pa_list, pb_list, joint_pdf):
    system_mutual_info = 0
    for i in range(0, len(joint_pdf)):
        for j in range(0, len(joint_pdf[i])):
            if abs(joint_pdf[i][j]) > 0 and abs(pa_list[i]) > 0 and abs(pb_list[j]) > 0:
                system_mutual_info = system_mutual_info + ( joint_pdf[i][j] * ( math.log2(joint_pdf[i][j] / (pa_list[i]*pb_list[j]) ) ) )

    return system_mutual_info

cond_prob = get_cond_prob(mat_file)
rows_sum_to_one = check_rows_sum_to_one(cond_prob)
if rows_sum_to_one:
    pa_list = create_initial_pa_list(cond_prob)
    pb_list = get_pb(cond_prob, pa_list)
    mutual_info_list = get_mutual_info_list(cond_prob, pa_list, pb_list)
    print("Initial Mutual Info List: ",mutual_info_list)
    variance = statistics.pvariance(mutual_info_list)
    print("Initial Mutual Info Variance: ", variance)
    if variance > 0.000001:
        # If first solution does not work, iterate until Mutual Information values are similar
        # n is the number of runs
        n = 0
        lowest_variance = 10000000
        while n < 15000:
            n = n + 1
            pa_list, increment = alter_pa_list(pa_list, mutual_info_list)
            pb_list = get_pb(cond_prob, pa_list)
            mutual_info_list = get_mutual_info_list(cond_prob, pa_list, pb_list)
            variance = statistics.pvariance(mutual_info_list)
            if variance < lowest_variance:
                lowest_mutual_info_list = copy.deepcopy(mutual_info_list)
                lowest_variance = variance
                lowest_n = n
        print("--------------------------------------")
        print("Lowest Variance:", lowest_variance)
        print("Lowest Mutual Information List: ", lowest_mutual_info_list)
        print("Total Runs: ", n)
        print("p(a) Increment Size: ", increment)
        mutual_info_list = lowest_mutual_info_list
    else:
        print("--------------------------------------")
        print("A uniform probability for p(a) was successful.")
        print("Lowest Variance:", variance)
        print("Lowest Mutual Information List: ", mutual_info_list)
    channel_capacity = get_channel_capacity(mutual_info_list)
    print("Channel Capacity: ", channel_capacity)
    source_entropy = get_source_entropy(pa_list, cond_prob)
    print("Source Entropy H(A): ", source_entropy)
    joint_pdf = get_joint_pdf(pa_list, cond_prob)
    inverse_cond_prob = get_inverse_cond_prob(pb_list, joint_pdf)
    sink_entropy = get_source_entropy(pb_list, inverse_cond_prob)
    print("Sink Entropy H(B): ", sink_entropy)
    conditional_entropy_given_A = get_conditional_entropy(joint_pdf, cond_prob, False)
    print("Conditional Entropy H(B|A): ", conditional_entropy_given_A)
    conditional_entropy_given_B = get_conditional_entropy(joint_pdf, inverse_cond_prob, True)
    print("Conditional Entropy H(A|B): ", conditional_entropy_given_B)
    equivocation = get_equivocation(joint_pdf)
    print("Equivocation H(A,B): ", equivocation)
    system_mutual_info = get_system_mutual_info(pa_list, pb_list, joint_pdf)
    print("System Mutual Information I(A;B): ", system_mutual_info)



