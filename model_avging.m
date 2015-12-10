% PUT THE NAMES OF THE RESULTS FILES TO RUN HERE
files = [fopen('nets/jakenet-aug8flip-refnet3/test-predictions-30.txt'), ...
    fopen('nets/jakenet-aug4flip/test-predictions-30.txt'), ...
    fopen('nets/jakenet/test-predictions-20.txt'), ...
    fopen('nets/alexnet1/test-predictions-21.txt'), ...
    fopen('nets/jamar7/test-predictions-20.txt'), ...
    fopen('nets/jakenet-flipNoise/test-predictions-21.txt')];
num_models = length(files);

output_file_id = fopen('avg_results.txt','w');
formatSpec = '%s %d %d %d %d %d\n';

% WHETHER RUNNING ON VAL OR TEST SET, CHANGE THIS TO CHANGE WHAT RUNNING ON
run_on_val_set = false;

not_done = true;

while not_done

    % Array storing the indices
    indices = 101*ones(num_models*5,1);
    % Parallel array storing the number of occurrences of each index
    num_occ = zeros(num_models*5,1);
    counter = 1;
    for i=1:num_models
        tline = fgetl(files(i));
        line_chars = strsplit(tline);
        
        % Grabbing the filename
        if i == 1
            prefix = cell2mat(line_chars(1));
            
            if run_on_val_set
                if strcmp('val/00010000.jpg',prefix)
                    not_done = false;
                end
            else
                if strcmp('test/00010000.jpg',prefix)
                    not_done = false;
                end
            end
        end
        
        % Adding the indices
        for j=2:length(line_chars)
            val = str2double(cell2mat(line_chars(j)));
            val_index = find(indices == val);
            %             If the value does not exist already, add it to
            %             the indices
            if length(val_index) == 0
                indices(counter) = val;
                num_occ(counter) = 1;
                counter = counter + 1;
            else
                num_occ(val_index) = num_occ(val_index) + 1;
            end
        end
    end

    % sort the values
    [sorted_num_occ, idx_sort] = sort(num_occ,'descend');
    
    % Write output
    fprintf(output_file_id,formatSpec, prefix, indices(idx_sort(1)), ...
            indices(idx_sort(2)), indices(idx_sort(3)), ...
            indices(idx_sort(4)), indices(idx_sort(5)));
end

fclose(output_file_id);