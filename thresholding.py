# Copyright (c) 2024 Jonathan S. Pollack (https://github.com/JPPhoto)

import numpy
from scipy.ndimage import gaussian_filter1d

from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    ImageField,
    InputField,
    InvocationContext,
    OutputField,
    WithBoard,
    WithMetadata,
    invocation,
    invocation_output,
)


@invocation_output("thresholding_output")
class ThresholdingOutput(BaseInvocationOutput):
    """Thresholding output class"""

    highlights_mask: ImageField = OutputField()
    midtones_mask: ImageField = OutputField()
    shadows_mask: ImageField = OutputField()


@invocation("thresholding", title="Thresholding", tags=["thresholding"], version="1.1.3")
class ThresholdingInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Puts out 3 masks for a source image representing highlights, midtones, and shadows"""

    image: ImageField = InputField(description="The image to add film grain to")
    highlights_point: int = InputField(ge=0, le=255, description="Highlight point", default=170)
    shadows_point: int = InputField(ge=0, le=255, description="Shadow point", default=85)
    lut_blur: float = InputField(ge=0, description="LUT blur", default=0.0)

    def gaussian_blur(self, data):
        if self.lut_blur == 0.0:
            return data
        else:
            arr = numpy.asarray(data)
            filtered_data = gaussian_filter1d(input=arr, sigma=self.lut_blur)
            return filtered_data.tolist()

    def invoke(self, context: InvocationContext) -> ThresholdingOutput:
        image = context.images.get_pil(self.image.image_name, mode="L")

        highlights_lut = [0 if p > self.highlights_point else 255 for p in range(0, 256)]
        midtones_lut = [0 if (p <= self.highlights_point and p > self.shadows_point) else 255 for p in range(0, 256)]
        shadows_lut = [0 if p <= self.shadows_point else 255 for p in range(0, 256)]

        highlights_lut = self.gaussian_blur(highlights_lut)
        midtones_lut = self.gaussian_blur(midtones_lut)
        shadows_lut = self.gaussian_blur(shadows_lut)

        highlights_mask = image.point(lambda p: highlights_lut[p])
        midtones_mask = image.point(lambda p: midtones_lut[p])
        shadows_mask = image.point(lambda p: shadows_lut[p])

        h_image_dto = context.images.save(image=highlights_mask)
        m_image_dto = context.images.save(image=midtones_mask)

        s_image_dto = context.images.save(image=shadows_mask)

        highlights_output = ImageField(image_name=h_image_dto.image_name)

        midtones_output = ImageField(image_name=m_image_dto.image_name)

        shadows_output = ImageField(image_name=s_image_dto.image_name)

        return ThresholdingOutput(
            highlights_mask=highlights_output,
            midtones_mask=midtones_output,
            shadows_mask=shadows_output,
        )
