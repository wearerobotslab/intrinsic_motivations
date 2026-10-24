// Minimal stand-in for AREFramework's are::Control, just enough to build and
// test the homeokinesis controller outside of ARE.
#ifndef ARE_CONTROL_STUB_H
#define ARE_CONTROL_STUB_H
#include <memory>
#include <vector>

namespace are {
class Control {
public:
    typedef std::shared_ptr<Control> Ptr;
    typedef std::shared_ptr<const Control> ConstPtr;
    Control() = default;
    Control(const Control&) = default;
    virtual ~Control() = default;
    virtual Ptr clone() const = 0;
    virtual std::vector<double> update(const std::vector<double>& sensorValues) = 0;
};
} // namespace are
#endif
