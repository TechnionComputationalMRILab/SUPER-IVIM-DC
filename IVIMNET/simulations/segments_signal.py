import SimpleITK as sitk


def segments_signal(file_path, slice_number):

    seg_slice = []
    reader = sitk.ImageFileReader()
    reader.SetFileName(file_path)
    image = reader.Execute()
    nda = sitk.GetArrayFromImage(image)
    if nda.ndim == 2:
        seg_slice.append(nda)
    elif nda.ndim == 3:
        seg_slice.append(nda[slice_number, :, :])
    else:
        assert (nda.ndim == 2), "Error in image dimensions. Dimensions of image in file must be 2 or 3."

    return seg_slice
