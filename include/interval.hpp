#ifndef INTERVAL_HPP
#define INTERVAL_HPP

#define IS_OVERLAPPING(interval_1, interval_2) (interval_2.min <= interval_1.max && interval_1.min <= interval_2.max)

class Interval {

    public:

        double min;
        double max;
};

#endif