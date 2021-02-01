import unittest
import pycg.pycg as cg
import numpy as np


class WorldObjectTestCase(unittest.TestCase):
    """
    a base class whose setup method creates a new world object
    """

    def setUp(self):
        self._obj = cg.WorldObject()


class TestHomogeneousCoordinate(unittest.TestCase):
    def setUp(self):
        self.coord = cg.HomogeneousCoordinate(3, 4, 5, 6)

    def test_getting_object_values(self):
        getters = [getattr(self.coord, getter) for getter in "xyzw"]
        for n, getter in enumerate(getters):
            self.assertEqual(self.coord[n], getter)

    def test_setting_object_values(self):
        """
        we should be able to update the coordinate entries by both array indexing and value
        :return:
        """
        # update by calling the setter method
        value_to_set = 0
        [setattr(self.coord, setter, value_to_set) for setter in "xyzw"]
        for n, getter in enumerate("xyzw"):
            self.assertEqual(getattr(self.coord, getter), value_to_set)
            self.assertEqual(self.coord[n], value_to_set)

        # update by directly calling the array index
        value_to_set = 1000
        for n in range(4):
            self.coord[n] = value_to_set

        for n, getter in enumerate("xyzw"):
            self.assertEqual(getattr(self.coord, getter), value_to_set)
            self.assertEqual(self.coord[n], value_to_set)


class TestPoint(unittest.TestCase):
    def setUp(self):
        self.coord = cg.Point(3, 4, 5)

    def test_fields_accessable(self):
        self.assertEqual(self.coord.w, 1)


class TestVector(unittest.TestCase):
    def setUp(self):
        self.coord = cg.Vector(3, 4, 5)

    def test_fields_accessable(self):
        self.assertEqual(self.coord.w, 0)


class TestWorldObjectCreation(WorldObjectTestCase):
    def test_object_creation(self):
        # the object should be centered at the origin facing the positive z-axis
        self.assertTrue(np.array_equal(self._obj.get_position(), cg.Point(0, 0, 0)))
        self.assertTrue(np.array_equal(self._obj.get_orientation(), cg.Vector(0, 0, 1)))


class TestWorldObjectScaling(WorldObjectTestCase):
    def setUp(self):
        super().setUp()  # call the parent setup function to make the object
        self._obj.move(1, 1, 1)  # move the object to 1,1,1 so scale operations work

    def test_orientation_does_not_scale(self):
        self._obj.scale(100, 100, 100)
        self.assertTrue(np.allclose(self._obj.get_orientation(), cg.Vector(0, 0, 1.)))

    def test_single_axis_scale(self):
        # move the object to 1,1,1 so that the scale effects it
        scale_axes = "xyz"
        scale_fns = [getattr(self._obj, "scale_" + axis) for axis in scale_axes]
        scale_values = [3, 4, 5]
        for fn, scale in zip(scale_fns, scale_values):
            fn(scale)
        self.assertTrue(np.allclose(self._obj.get_position(), cg.Point(*scale_values)), f"{self._obj.get_position()}")

    def test_3axis_scale(self):
        scale_values = (3, 4, 5)
        self._obj.scale(*scale_values)
        self.assertTrue(np.allclose(self._obj.get_position(), cg.Point(*scale_values)), f"{self._obj.get_position()}")

    def test_chained_scale(self):
        scale_values = (3, 4, 5)
        self._obj.scale_x(scale_values[0]).scale_y(scale_values[1]).scale_z(scale_values[2])
        self.assertTrue(np.allclose(self._obj.get_position(), cg.Point(*scale_values)), f"{self._obj.get_position()}")

    def test_scale_all(self):
        scale_factor = 1000
        expected_pos = scale_factor*np.ones(3)
        self._obj.scale_all(scale_factor)
        self.assertTrue(np.allclose(self._obj.get_position(), cg.Point(*expected_pos)), f"{self._obj.get_position()}")


class TestWorldObjectTranslation(WorldObjectTestCase):
    def test_3axis_movement(self):
        my_obj = cg.WorldObject()

        # We should be able to move the object multiple times and the position will move but not direction
        move_vector = np.array((1, 2, -5))
        my_obj.move(*move_vector)  # move the object in space
        self.assertTrue(np.array_equal(my_obj.get_position(), cg.Point(*move_vector)))

        # reversing the move gets you back to the origin
        my_obj.move(*(-move_vector))  # move the object in space
        self.assertTrue(np.array_equal(my_obj.get_position(), cg.Point()))

    def test_single_axis_movement(self):
        # individual move functions execute properly
        attr_names = ["move_" + direction for direction in "xyz"]
        move_attrs = [getattr(self._obj, attribute) for attribute in attr_names]
        movement = 3
        for n, fn_call in enumerate(move_attrs):
            fn_call(movement)
            self.assertEqual(self._obj.get_position()[n], movement)

    def test_chained_movement(self):
        movement = 3
        self._obj.move_x(movement).move_y(movement).move_z(movement)
        self.assertTrue(np.array_equal(self._obj.get_position(), cg.Point(movement, movement, movement)))


class TestWorldObjectRotation(WorldObjectTestCase):
    def test_rotation(self):
        my_obj = self._obj

        # rotation about the y-axis by 90 degree should change the direction vector to x
        my_obj.rotate_y(90, units="deg")
        self.assertTrue(np.allclose(my_obj.get_orientation(), cg.Vector(1., 0, 0)))

        # now rotation it about the z-axis 90 degree should have it point to positive y
        my_obj.rotate_z(90, units="deg")
        self.assertTrue(np.allclose(my_obj.get_orientation(), cg.Vector(0, 1., 0)))

        # rotation 90 degree about the x-axis should reset it to positive z
        my_obj.rotate_x(90, units="deg")
        self.assertTrue(np.allclose(my_obj.get_orientation(), cg.Vector(0, 0, 1.)))

    def test_rotation_chain(self):
        # rotations should be able to be cascaded
        self._obj.rotate_y(90, units="deg").rotate_z(90, units="deg").rotate_x(90, units="deg")
        self.assertTrue(np.allclose(self._obj.get_orientation(), cg.Vector(0, 0, 1.)))

    def test_rotation_units(self):
        # rotations should work for radians and degrees
        rotation_angles = [90, np.pi / 2]
        rotation_units = ["deg", "rad"]
        for angle, unit in zip(rotation_angles, rotation_units):
            self._obj.rotate_y(angle, units=unit).rotate_z(angle, units=unit).rotate_x(angle, units=unit)
            self.assertTrue(np.allclose(self._obj.get_orientation(), cg.Vector(0, 0, 1.)),
                            f"Test Failed for unit {unit}, has orientation {self._obj.get_orientation()}")


if __name__ == '__main__':
    unittest.main()
